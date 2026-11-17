from __future__ import annotations

import sys
from pathlib import Path

import sympy

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.cost_model import PIPELINES
from src.cost_pass import run_cost_pass
from src.mlir_sympy import parse_cost_function
from src.parse import cost_equations


def expect_equations(
    actual: dict[str, sympy.Expr], expected: dict[str, sympy.Expr]
) -> None:
    assert tuple(actual) == PIPELINES
    for category in PIPELINES:
        assert sympy.simplify(actual[category] - expected.get(category, 0)) == 0, (
            category, actual[category]
        )


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: verify_work_units_pass.py TRITON_OPT PASS_PLUGIN SCALAR_TTGIR MATMUL_TTGIR"
        )
    triton_opt, plugin, scalar_fixture, matmul_fixture = map(Path, sys.argv[1:])
    scalar_mlir = run_cost_pass(
        triton_opt, plugin, scalar_fixture, "scalar_work", timeout_s=10.0
    )
    symbol = sympy.Symbol
    expect_equations(cost_equations(scalar_mlir), {
        "fp32": 128 * (
            symbol("arith.addf32") + symbol("arith.mulf32") + symbol("arith.fma32")
        ),
        "fp64": 128 * (symbol("arith.addf64") + symbol("arith.fma64")),
        "int32": 128 * (
            symbol("arith.addi") + symbol("arith.muli") + symbol("arith.cmpi")
            + symbol("triton.make_range_cost") + symbol("triton.addptr_cost")
        ),
        "memory": 512 * (symbol("triton.load_cost") + symbol("triton.store_cost")),
    })
    cost_function = parse_cost_function(scalar_mlir)
    attrs = cost_function.operation.result_attrs
    units = [attr["cost.unit"].value for attr in attrs]
    assert units == ["instructions"] * 4 + ["flops", "bytes", "bytes"]

    matmul_mlir = run_cost_pass(
        triton_opt, plugin, matmul_fixture, "tiled_matmul_kernel", timeout_s=10.0
    )
    # A 128 x 64 output tile accumulates over K=256; each MAC is two FLOPs.
    tensor = cost_equations(matmul_mlir)["tensor"]
    expected_tensor = 2 * 128 * 64 * 256 * symbol("triton.dot_cost")
    assert sympy.simplify(tensor - expected_tensor) == 0


if __name__ == "__main__":
    main()
