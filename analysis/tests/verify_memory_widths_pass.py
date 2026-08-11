from __future__ import annotations

import sys
from pathlib import Path

import sympy

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.cost_pass import run_cost_pass
from src.parse import cost_equations


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_memory_widths_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )

    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    cost_mlir = run_cost_pass(
        triton_opt,
        plugin,
        ttgir,
        "memory_element_widths",
        timeout_s=10.0,
    )
    memory = cost_equations(cost_mlir)["memory"]

    # One load and store of i8, f16, f32, and f64 transfers 15 bytes per
    # thread. The fixture runs 32 threads per block.
    expected = 480 * (
        sympy.Symbol("triton.load_cost") + sympy.Symbol("triton.store_cost")
    )
    if sympy.simplify(memory - expected) != 0:
        raise SystemExit(f"unexpected memory byte expression: {memory}")


if __name__ == "__main__":
    main()
