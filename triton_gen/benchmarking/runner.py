from dataclasses import asdict

import torch
import triton
from tqdm import tqdm
from triton.runtime.errors import OutOfResources

from . import storage
from .records import KernelRunRecord, collect_case_metadata, record_name


def benchmark_kernel(kernel, grid, args, kwargs, warmup_ms, rep_ms):
    compiled_kernel = kernel[grid](*args, **kwargs)
    torch.cuda.synchronize()

    times = triton.testing.do_bench(
        lambda: kernel[grid](*args, **kwargs),
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.2, 0.5, 0.8],
    )
    time_p20_ms, time_ms, time_p80_ms = (float(value) for value in times)
    spread = max(time_p80_ms - time_p20_ms, 0.0)
    time_cv = spread / time_ms if time_ms > 0 else 0.0
    return compiled_kernel, time_ms, time_p20_ms, time_p80_ms, time_cv


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _show_resume_position(progress, kernel_count, kernel_index, case_index):
    if kernel_index < kernel_count and (kernel_index or case_index):
        progress.write(
            f"Resuming at kernel {kernel_index + 1}/{kernel_count}, "
            f"case {case_index + 1}"
        )
    elif kernel_index == kernel_count:
        progress.write("Resuming final result write")


def run_sweep(kernel_modules, *, warmup_ms, rep_ms, restart=False):
    if restart:
        storage.RESUME_PATH.unlink(missing_ok=True)

    device = triton.runtime.driver.active.get_active_torch_device()
    kernel_names = [module.KERNEL.__name__ for module in kernel_modules]
    resume_state = storage.load_resume_state(kernel_names, warmup_ms, rep_ms)
    next_kernel_index = resume_state["next_kernel_index"]
    next_case_index = resume_state["next_case_index"]
    result = resume_state["result"]

    def save_progress(kernel_index, case_index):
        storage.save_resume_state(
            result,
            kernel_names,
            warmup_ms,
            rep_ms,
            kernel_index,
            case_index,
        )

    progress = tqdm(
        kernel_modules[next_kernel_index:],
        total=len(kernel_modules),
        initial=next_kernel_index,
        desc="Kernels",
        unit="kernel",
        dynamic_ncols=True,
    )
    _show_resume_position(
        progress,
        len(kernel_modules),
        next_kernel_index,
        next_case_index,
    )

    for kernel_index, module in enumerate(progress, start=next_kernel_index):
        kernel = module.KERNEL
        kernel_name = kernel.__name__
        kernel_runs = result.setdefault(kernel_name, [])
        resume_case_index = next_case_index if kernel_index == next_kernel_index else 0
        if len(kernel_runs) != resume_case_index:
            raise RuntimeError(
                f"Resume state has {len(kernel_runs)} records for {kernel_name}, "
                f"expected {resume_case_index}; use --restart to discard it"
            )
        progress.set_description(f"Kernel: {kernel_name}")

        case_count = 0
        for case_index, (args, kwargs, grid) in enumerate(module.iter_args(device)):
            case_count = case_index + 1
            metadata = collect_case_metadata(kernel, args, kwargs, grid)

            if case_index < resume_case_index:
                saved_record = kernel_runs[case_index]
                if any(saved_record.get(key) != value for key, value in metadata.items()):
                    raise RuntimeError(
                        f"Case {case_index + 1} for {kernel_name} changed since "
                        "the checkpoint; use --restart to start over"
                    )
                continue

            name_suffix = record_name(args, kwargs)
            config_name = f"{kernel_name}_{name_suffix}"
            try:
                benchmark = benchmark_kernel(
                    kernel,
                    grid,
                    args,
                    kwargs,
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                )
            except (OutOfResources, torch.cuda.OutOfMemoryError) as exc:
                run_record = KernelRunRecord(
                    **metadata,
                    compiled_name=None,
                    ttgir_filename=None,
                    time_ms=None,
                    time_p20_ms=None,
                    time_p80_ms=None,
                    time_cv=None,
                    status="skipped",
                    error=f"{type(exc).__name__}: {exc}",
                )
                kernel_runs.append(asdict(run_record))
                progress.write(config_name)
                progress.write(f"skipped: {type(exc).__name__}: {exc}\n")
                save_progress(kernel_index, case_index + 1)
                continue
            finally:
                clear_cuda_cache()

            compiled_kernel, elapsed_ms, time_p20_ms, time_p80_ms, time_cv = benchmark
            launch_name = f"{compiled_kernel.name}_{name_suffix}"
            ttgir_filename = storage.write_ttgir(
                launch_name,
                compiled_kernel.asm["ttgir"],
            )
            run_record = KernelRunRecord(
                **metadata,
                compiled_name=compiled_kernel.name,
                ttgir_filename=ttgir_filename,
                time_ms=elapsed_ms,
                time_p20_ms=time_p20_ms,
                time_p80_ms=time_p80_ms,
                time_cv=time_cv,
            )
            kernel_runs.append(asdict(run_record))

            progress.write(launch_name)
            progress.write(
                f"kernel time: {elapsed_ms:.6f} ms "
                f"(p20={time_p20_ms:.6f}, p80={time_p80_ms:.6f}, "
                f"spread/median={time_cv:.3f})\n"
            )
            save_progress(kernel_index, case_index + 1)

        if resume_case_index > case_count:
            raise RuntimeError(
                f"Resume state points to case {resume_case_index + 1}, "
                f"but {kernel_name} has only {case_count} cases"
            )

        next_case_index = 0
        save_progress(kernel_index + 1, 0)

    storage.write_result(result)
    storage.RESUME_PATH.unlink(missing_ok=True)
