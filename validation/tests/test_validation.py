from __future__ import annotations

import subprocess
import sys

import pytest

from main import cost_equation


SAMPLE_COST_MLIR = """// Cost expression for func.func @main
func.func @__cost_expr(%dpas_cost: f64 {cost.name = "dpas_cost"}) -> f64 {
  %cst = arith.constant 2.200000e+01 : f64
  %cst_0 = arith.constant 3.100000e+01 : f64
  %cst_1 = arith.constant 1.280000e+02 : f64
  %cst_2 = arith.constant 0.000000e+00 : f64
  %cst_3 = arith.constant 2.700000e+01 : f64
  %0 = arith.addf %dpas_cost, %cst_3 : f64
  %1 = arith.mulf %dpas_cost, %cst_0 : f64
  %2 = arith.addf %0, %1 : f64
  %3 = arith.addf %2, %cst_2 : f64
  %4 = arith.mulf %3, %cst_1 : f64
  %5 = arith.addf %4, %cst_3 : f64
  %6 = arith.addf %5, %cst : f64
  %7 = arith.addf %6, %cst_2 : f64
  return %7 : f64
}"""


def test_simplifies_generated_cost_expression():
    assert str(cost_equation(SAMPLE_COST_MLIR)) == "4096.0*dpas_cost + 3505.0"


def test_handles_multiple_variables_and_repeated_terms():
    mlir = """func.func @__cost_expr(%x: f64 {cost.name = "x"}, %y: f64 {cost.name = "y"}) -> f64 {
      %0 = arith.addf %x, %y : f64
      %1 = arith.mulf %0, %x : f64
      %2 = arith.addf %1, %x : f64
      return %2 : f64
    }"""

    assert str(cost_equation(mlir)) == "x*(x + y + 1)"


def test_preserves_dotted_variable_names():
    mlir = """func.func @__cost_expr(%residency.p1: f64 {cost.name = "residency.p1"}, %residency.l1_cost: f64 {cost.name = "residency.l1_cost"}) -> f64 {
      %0 = arith.mulf %residency.p1, %residency.l1_cost : f64
      return %0 : f64
    }"""

    assert str(cost_equation(mlir)) == "residency.l1_cost*residency.p1"


def test_rejects_unsupported_operation():
    mlir = """func.func @__cost_expr(%x: f64 {cost.name = "x"}) -> f64 {
      %0 = arith.divf %x, %x : f64
      return %0 : f64
    }"""

    with pytest.raises(ValueError, match="unsupported op"):
        cost_equation(mlir)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="input MLIR is empty"):
        cost_equation("")


def test_cli_reports_validation_errors_without_traceback():
    result = subprocess.run(
        [sys.executable, "main.py", "cost.mlir"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "error: input MLIR is empty\n"
    assert "Traceback" not in result.stderr


def test_cli_reads_stdin():
    result = subprocess.run(
        [sys.executable, "main.py", "-"],
        input=SAMPLE_COST_MLIR,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "4096.0*dpas_cost + 3505.0"
