from __future__ import annotations

import sys
from pathlib import Path

import sympy

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.cost_pass import run_cost_pass
from src.parse import cost_equations


def l1_equation(triton_opt: Path, plugin: Path, ttgir: Path, function: str):
    cost_mlir = run_cost_pass(
        triton_opt,
        plugin,
        ttgir,
        function,
        timeout_s=10.0,
    )
    return cost_equations(cost_mlir)["l1"]


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: verify_l1_widths_pass.py TRITON_OPT PASS_PLUGIN TTGIR")

    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    local_alloc = sympy.Symbol("triton_gpu.local_alloc_cost")
    local_load = sympy.Symbol("triton_gpu.local_load_cost")

    # Each function moves 32 elements per thread across a 32-thread block.
    for function, element_bytes in (("shared_f16", 2), ("shared_f32", 4)):
        actual = l1_equation(triton_opt, plugin, ttgir, function)
        expected = 1024 * element_bytes * (local_alloc + local_load)
        if sympy.simplify(actual - expected) != 0:
            raise SystemExit(f"unexpected {function} L1 byte expression: {actual}")

    alloc_only = l1_equation(triton_opt, plugin, ttgir, "shared_alloc_only")
    if alloc_only.is_zero is not True:
        raise SystemExit(
            f"source-free local allocation produced L1 traffic: {alloc_only}"
        )


if __name__ == "__main__":
    main()
