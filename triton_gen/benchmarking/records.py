from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch


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


def record_name(args, kwargs):
    arg_parts = [str(arg) for arg in args if not isinstance(arg, torch.Tensor)]
    kwarg_parts = [f"{key}={value}" for key, value in sorted(kwargs.items())]
    return "_".join([*arg_parts, *kwarg_parts])


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


def collect_case_metadata(kernel, args, kwargs, grid):
    return {
        "args": [str(arg) for arg in args if not isinstance(arg, torch.Tensor)],
        "kwargs": {key: str(value) for key, value in sorted(kwargs.items())},
        "scalar_args": record_scalar_args(kernel, args, kwargs),
        "grid_size": record_grid_size(grid, kwargs),
        "block_size": record_block_size(kwargs),
    }
