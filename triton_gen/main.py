from dataclasses import asdict, dataclass
import json
import os
import re

import torch
import triton
import vec_add_kernel


DEVICE = triton.runtime.driver.active.get_active_torch_device()
KERNEL_MODULES = [vec_add_kernel]

@dataclass
class KernelRunRecord:
    args: list[str]
    kwargs: dict[str, str]
    compiled_name: str
    ttgir_filename: str
    time_ms: float


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


if __name__ == "__main__":
    result = {}

    for module in KERNEL_MODULES:
        kernel = module.KERNEL
        kernel_name = kernel.__name__
        kernel_runs = []

        for args, kwargs, grid in module.iter_args(DEVICE):
            kernel[grid](*args, **kwargs)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            h = kernel[grid](*args, **kwargs)
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end)

            # save the kernel module
            launch_name = f"{h.name}_{record_name(args, kwargs)}"
            ttgir_filename = write_ttgir(launch_name, h.asm["ttgir"])

            runRecord = KernelRunRecord(
                args=[str(arg) for arg in args if not isinstance(arg, torch.Tensor)],
                kwargs={key: str(value) for key, value in sorted(kwargs.items())},
                compiled_name=h.name,
                ttgir_filename=ttgir_filename,
                time_ms=elapsed_ms,
            )
            kernel_runs.append(runRecord)

            print(launch_name)
            print(f"kernel time: {elapsed_ms:.3f} ms")
            print(h.asm["ttgir"])
            print(result)

        result[kernel_name] = [asdict(kernel_run) for kernel_run in kernel_runs]

    write_result(result)
