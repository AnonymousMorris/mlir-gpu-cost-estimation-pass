from dataclasses import asdict, dataclass
import argparse
import json
import os
import re

import torch
import triton
from kernels import KERNEL_MODULES


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@dataclass
class KernelRunRecord:
    args: list[str]
    kwargs: dict[str, str]
    compiled_name: str
    ttgir_filename: str
    time_ms: float
    time_p20_ms: float
    time_p80_ms: float
    time_cv: float


def write_result(result):
    os.makedirs("results", exist_ok=True)
    with open("results/result.json", "w") as fs:
        json.dump(result, fs, indent=2)


def record_name(args, kwargs):
    arg_parts = [str(arg) for arg in args if not isinstance(arg, torch.Tensor)]
    kwarg_parts = [f"{key}={value}" for key, value in sorted(kwargs.items())]
    return "_".join([*arg_parts, *kwarg_parts])


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)


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
    cli_args = parser.parse_args()

    result = {}

    for module in KERNEL_MODULES:
        kernel = module.KERNEL
        kernel_name = kernel.__name__
        kernel_runs = []

        for kernel_args, kwargs, grid in module.iter_args(DEVICE):
            h, elapsed_ms, time_p20_ms, time_p80_ms, time_cv = benchmark_kernel(
                kernel,
                grid,
                kernel_args,
                kwargs,
                warmup_ms=cli_args.warmup_ms,
                rep_ms=cli_args.rep_ms,
            )

            # save the kernel module
            launch_name = f"{h.name}_{record_name(kernel_args, kwargs)}"
            ttgir_filename = write_ttgir(launch_name, h.asm["ttgir"])

            runRecord = KernelRunRecord(
                args=[str(arg) for arg in kernel_args if not isinstance(arg, torch.Tensor)],
                kwargs={key: str(value) for key, value in sorted(kwargs.items())},
                compiled_name=h.name,
                ttgir_filename=ttgir_filename,
                time_ms=elapsed_ms,
                time_p20_ms=time_p20_ms,
                time_p80_ms=time_p80_ms,
                time_cv=time_cv,
            )
            kernel_runs.append(runRecord)

            print(launch_name)
            print(
                f"kernel time: {elapsed_ms:.6f} ms "
                f"(p20={time_p20_ms:.6f}, p80={time_p80_ms:.6f}, spread/median={time_cv:.3f})\n"
            )
            # print(h.asm["ttgir"])
            # print(result)

        result[kernel_name] = [asdict(kernel_run) for kernel_run in kernel_runs]

    write_result(result)
