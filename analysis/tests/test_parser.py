from __future__ import annotations

import pytest

from cost_analysis.parser import CostParseError, parse_cost_equation


SAMPLE_MLIR = """// Cost expression for tt.func @vec_add
func.func @__cost_expr(%triton.load_cost: f64 {cost.name = "triton.load_cost"}, %triton.store_cost: f64 {cost.name = "triton.store_cost"}) -> f64 {
  %cst = arith.constant 2.000000e+00 : f64
  %0 = arith.mulf %triton.load_cost, %cst : f64
  %1 = arith.addf %0, %triton.store_cost : f64
  %2 = arith.addf %1, %cst : f64
  return %2 : f64
}"""


def test_parse_linear_cost_equation():
    equation = parse_cost_equation(SAMPLE_MLIR)

    assert equation.variables == ["triton.load_cost", "triton.store_cost"]
    assert equation.coefficients == {
        "triton.load_cost": 2.0,
        "triton.store_cost": 1.0,
    }
    assert equation.constant == 2.0


def test_binds_scalar_kernel_argument_cost_names():
    mlir = """func.func @__cost_expr(%_7: i32 {cost.name = "7"}) -> f64 {
      %0 = arith.uitofp %_7 : i32 to f64
      return %0 : f64
    }"""

    equation = parse_cost_equation(mlir, scalar_values={"7": 512})

    assert equation.constant == 512.0
    assert equation.variables == []


def test_rejects_unbound_scalar_kernel_argument_cost_names():
    mlir = """func.func @__cost_expr(%n_rows: i32 {cost.name = "n_rows"}) -> f64 {
      %0 = arith.uitofp %n_rows : i32 to f64
      return %0 : f64
    }"""

    with pytest.raises(CostParseError, match="missing scalar value"):
        parse_cost_equation(mlir)
