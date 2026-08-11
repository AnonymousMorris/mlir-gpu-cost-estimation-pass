from __future__ import annotations

import sys
from pathlib import Path

import sympy

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.cost_pass import run_cost_pass
from src.parse import cost_equations


def fp32_equation(
    triton_opt: Path,
    plugin: Path,
    ttgir: Path,
    function: str,
) -> sympy.Expr:
    cost_mlir = run_cost_pass(
        triton_opt,
        plugin,
        ttgir,
        function,
        timeout_s=10.0,
    )
    return cost_equations(cost_mlir)["fp32"]


def expect_equation(actual: sympy.Expr, expected: sympy.Expr, function: str) -> None:
    if sympy.simplify(actual - expected) != 0:
        raise SystemExit(f"unexpected {function} FP32 expression: {actual}")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_reduction_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )

    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    addi = sympy.Symbol("arith.addi")
    addf32 = sympy.Symbol("arith.addf32")
    shuffle = sympy.Symbol("triton.reduce_shuffle_cost")

    expected = {
        # Four values per thread need three combines and no warp shuffles.
        "reduce_thread_only": 384 * addi,
        # One accumulator crosses 32 lanes in five shuffle/combine stages.
        "reduce_warp_only": 640 * addf32 + 640 * shuffle,
        # Two accumulators each need three local combines and three warp stages.
        "reduce_thread_and_warp": 1536 * addi + 768 * shuffle,
        # Four warps contain unique axis data. The inter-warp stub deliberately
        # adds nothing, leaving only the two implemented hierarchy levels.
        "reduce_inter_warp_stub": 512 * addi + 384 * shuffle,
        # This is Triton's nontrivial linear-layout reduction lowering fixture.
        "reduce_linear_layout": 1280 * addi + 1024 * shuffle,
        # The combiner region is shared, while each source value is shuffled.
        "reduce_multi_result": (
            1536 * addf32 + 1536 * addi + 1536 * shuffle
        ),
    }

    for function, expected_equation in expected.items():
        actual = fp32_equation(triton_opt, plugin, ttgir, function)
        expect_equation(actual, expected_equation, function)


if __name__ == "__main__":
    main()
