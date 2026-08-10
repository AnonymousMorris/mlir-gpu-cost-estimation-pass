from dataclasses import asdict, dataclass
import argparse
import json
import math
from numbers import Integral, Real
import os
import re
from pathlib import Path

import torch
import triton
from triton.runtime.errors import OutOfResources
from tqdm import tqdm
from kernels import KERNEL_MODULES


RESULT_PATH = Path("results/result.json")
RESUME_PATH = Path("results/run_state.json.tmp")
RESUME_VERSION = 2

@dataclass
class KernelRunRecord:
    args: list[str]
    kwargs: dict[str, str]
    scalar_args: dict[str, int | float]
    grid_size: list[int]
    block_size: dict[str, str]
    compiled_name: str | None
    ttgir_filename: str | None
    time_ms: float | None
    time_p20_ms: float | None
    time_p80_ms: float | None
    time_cv: float | None
    status: str = "ok"
    error: str | None = None


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = path.with_name(f"{path.name}.new")
    with pending_path.open("w") as fs:
        json.dump(value, fs, indent=2)
        fs.flush()
        os.fsync(fs.fileno())
    os.replace(pending_path, path)


def write_result(result):
    atomic_write_json(RESULT_PATH, result)


def save_resume_state(result, kernel_names, warmup_ms, rep_ms, next_kernel_index, next_case_index):
    atomic_write_json(
        RESUME_PATH,
        {
            "version": RESUME_VERSION,
            "kernel_names": kernel_names,
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            "next_kernel_index": next_kernel_index,
            "next_case_index": next_case_index,
            "result": result,
        },
    )


def load_resume_state(kernel_names, warmup_ms, rep_ms):
    if not RESUME_PATH.exists():
        return {
            "next_kernel_index": 0,
            "next_case_index": 0,
            "result": {},
        }

    try:
        state = json.loads(RESUME_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {RESUME_PATH}; use --restart to discard it") from exc

    expected = {
        "version": RESUME_VERSION,
        "kernel_names": kernel_names,
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
    }
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        fields = ", ".join(mismatches)
        raise RuntimeError(f"Resume state does not match this run ({fields}); use --restart to start over")

    next_kernel_index = state.get("next_kernel_index")
    next_case_index = state.get("next_case_index")
    result = state.get("result")
    valid_cursor = (
        isinstance(next_kernel_index, int)
        and 0 <= next_kernel_index <= len(kernel_names)
        and isinstance(next_case_index, int)
        and next_case_index >= 0
        and (next_kernel_index < len(kernel_names) or next_case_index == 0)
    )
    if not valid_cursor or not isinstance(result, dict):
        raise RuntimeError(f"Invalid resume state in {RESUME_PATH}; use --restart to discard it")

    return state


def record_name(args, kwargs):
    arg_parts = [str(arg) for arg in args if not isinstance(arg, torch.Tensor)]
    kwarg_parts = [f"{key}={value}" for key, value in sorted(kwargs.items())]
    return "_".join([*arg_parts, *kwarg_parts])


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)


def record_grid_size(grid, kwargs):
    grid_size = grid(kwargs) if callable(grid) else grid
    if isinstance(grid_size, int):
        return [grid_size]
    return [int(dimension) for dimension in grid_size]


def record_block_size(kwargs):
    return {
        key: str(value)
        for key, value in sorted(kwargs.items())
        if key.upper().startswith("BLOCK")
    }


def record_scalar_args(kernel, args, kwargs):
    argument_names = list(kernel.arg_names)
    if len(args) > len(argument_names):
        raise ValueError("kernel has more positional values than named arguments")

    arguments = dict(zip(argument_names, args))
    for name in argument_names:
        if name not in kwargs:
            continue
        if name in arguments:
            raise ValueError(f"kernel argument {name} was passed more than once")
        arguments[name] = kwargs[name]

    scalars = {}
    for name, value in arguments.items():
        if isinstance(value, bool):
            scalars[name] = int(value)
        elif isinstance(value, Integral):
            scalars[name] = int(value)
        elif isinstance(value, Real):
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"kernel argument {name} must be finite")
            scalars[name] = normalized
    return scalars


def write_ttgir(name, ttgir):
    os.makedirs("results/ttgir", exist_ok=True)
    filename = f"{safe_filename(name)}.ttgir"
    with open(f"results/ttgir/{filename}", "w") as fs:
        fs.write(ttgir)
    return filename


def benchmark_kernel(kernel, grid, args, kwargs, warmup_ms, rep_ms):
    h = kernel[grid](*args, **kwargs)
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
    return h, time_ms, time_p20_ms, time_p80_ms, time_cv


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Triton TTGIR files and benchmark timings.")
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=100,
        help="Approximate warmup duration per benchmark case.",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=1000,
        help="Approximate measured repetition duration per benchmark case.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=f"Discard {RESUME_PATH} and start from the first kernel.",
    )
    cli_args = parser.parse_args()

    if cli_args.restart:
        RESUME_PATH.unlink(missing_ok=True)

    device = triton.runtime.driver.active.get_active_torch_device()
    kernel_names = [module.KERNEL.__name__ for module in KERNEL_MODULES]
    resume_state = load_resume_state(kernel_names, cli_args.warmup_ms, cli_args.rep_ms)
    next_kernel_index = resume_state["next_kernel_index"]
    next_case_index = resume_state["next_case_index"]
    result = resume_state["result"]

    kernel_progress = tqdm(
        KERNEL_MODULES[next_kernel_index:],
        total=len(KERNEL_MODULES),
        initial=next_kernel_index,
        desc="Kernels",
        unit="kernel",
        dynamic_ncols=True,
    )
    if next_kernel_index < len(KERNEL_MODULES) and (next_kernel_index or next_case_index):
        kernel_progress.write(
            f"Resuming at kernel {next_kernel_index + 1}/{len(KERNEL_MODULES)}, case {next_case_index + 1}"
        )
    elif next_kernel_index == len(KERNEL_MODULES):
        kernel_progress.write("Resuming final result write")

    for kernel_index, module in enumerate(kernel_progress, start=next_kernel_index):
        kernel = module.KERNEL
        kernel_name = kernel.__name__
        kernel_runs = result.setdefault(kernel_name, [])
        resume_case_index = next_case_index if kernel_index == next_kernel_index else 0
        if len(kernel_runs) != resume_case_index:
            raise RuntimeError(
                f"Resume state has {len(kernel_runs)} records for {kernel_name}, expected {resume_case_index}; "
                "use --restart to discard it"
            )
        kernel_progress.set_description(f"Kernel: {kernel_name}")

        case_count = 0
        for case_index, (kernel_args, kwargs, grid) in enumerate(
            module.iter_args(device)
        ):
            case_count = case_index + 1
            run_args = [
                str(arg) for arg in kernel_args if not isinstance(arg, torch.Tensor)
            ]
            run_kwargs = {
                key: str(value) for key, value in sorted(kwargs.items())
            }
            scalar_args = record_scalar_args(kernel, kernel_args, kwargs)
            grid_size = record_grid_size(grid, kwargs)
            block_size = record_block_size(kwargs)

            if case_index < resume_case_index:
                saved_record = kernel_runs[case_index]
                current_metadata = {
                    "args": run_args,
                    "kwargs": run_kwargs,
                    "scalar_args": scalar_args,
                    "grid_size": grid_size,
                    "block_size": block_size,
                }
                if any(
                    saved_record.get(key) != value
                    for key, value in current_metadata.items()
                ):
                    raise RuntimeError(
                        f"Case {case_index + 1} for {kernel_name} changed since the checkpoint; "
                        "use --restart to start over"
                    )
                continue

            config_name = f"{kernel_name}_{record_name(kernel_args, kwargs)}"
            try:
                h, elapsed_ms, time_p20_ms, time_p80_ms, time_cv = benchmark_kernel(
                    kernel,
                    grid,
                    kernel_args,
                    kwargs,
                    warmup_ms=cli_args.warmup_ms,
                    rep_ms=cli_args.rep_ms,
                )
            except (OutOfResources, torch.cuda.OutOfMemoryError) as exc:
                run_record = KernelRunRecord(
                    args=run_args,
                    kwargs=run_kwargs,
                    scalar_args=scalar_args,
                    grid_size=grid_size,
                    block_size=block_size,
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
                kernel_progress.write(config_name)
                kernel_progress.write(f"skipped: {type(exc).__name__}: {exc}\n")
                save_resume_state(
                    result,
                    kernel_names,
                    cli_args.warmup_ms,
                    cli_args.rep_ms,
                    kernel_index,
                    case_index + 1,
                )
                continue
            finally:
                clear_cuda_cache()

            # save the kernel module
            launch_name = f"{h.name}_{record_name(kernel_args, kwargs)}"
            ttgir_filename = write_ttgir(launch_name, h.asm["ttgir"])

            runRecord = KernelRunRecord(
                args=run_args,
                kwargs=run_kwargs,
                scalar_args=scalar_args,
                grid_size=grid_size,
                block_size=block_size,
                compiled_name=h.name,
                ttgir_filename=ttgir_filename,
                time_ms=elapsed_ms,
                time_p20_ms=time_p20_ms,
                time_p80_ms=time_p80_ms,
                time_cv=time_cv,
            )
            kernel_runs.append(asdict(runRecord))

            kernel_progress.write(launch_name)
            kernel_progress.write(
                f"kernel time: {elapsed_ms:.6f} ms "
                f"(p20={time_p20_ms:.6f}, p80={time_p80_ms:.6f}, spread/median={time_cv:.3f})\n"
            )
            save_resume_state(
                result,
                kernel_names,
                cli_args.warmup_ms,
                cli_args.rep_ms,
                kernel_index,
                case_index + 1,
            )

        if resume_case_index > case_count:
            raise RuntimeError(
                f"Resume state points to case {resume_case_index + 1}, but {kernel_name} has only {case_count} cases"
            )

        next_case_index = 0
        save_resume_state(
            result,
            kernel_names,
            cli_args.warmup_ms,
            cli_args.rep_ms,
            kernel_index + 1,
            0,
        )

    write_result(result)
    RESUME_PATH.unlink(missing_ok=True)
