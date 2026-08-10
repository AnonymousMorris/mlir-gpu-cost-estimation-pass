from __future__ import annotations

import subprocess
import sys

import pytest

from src.parse import cost_equations


PARSE_MODULE = "src.parse"
CATEGORY_RESULTS = """(
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "l1"},
  f64 {cost.name = "memory"}
)"""


def category_cost_mlir(
    arguments: str,
    operations: str,
    fp32_result: str,
) -> str:
    return f"""func.func @__cost_expr({arguments}) -> {CATEGORY_RESULTS} {{
  %zero = arith.constant 0.000000e+00 : f64
{operations}
  return {fp32_result}, %zero, %zero, %zero, %zero, %zero : f64, f64, f64, f64, f64, f64
}}"""


SAMPLE_COST_MLIR = category_cost_mlir(
    '%dpas_cost: f64 {cost.name = "dpas_cost"}',
    """  %cst = arith.constant 2.200000e+01 : f64
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
  %7 = arith.addf %6, %cst_2 : f64""",
    "%7",
)


def test_simplifies_generated_cost_expression():
    equations = cost_equations(SAMPLE_COST_MLIR)

    assert str(equations["fp32"]) == "4096.0*dpas_cost + 3505.0"


def test_handles_multiple_variables_and_repeated_terms():
    mlir = category_cost_mlir(
        '%x: f64 {cost.name = "x"}, %y: f64 {cost.name = "y"}',
        """  %0 = arith.addf %x, %y : f64
  %1 = arith.mulf %0, %x : f64
  %2 = arith.addf %1, %x : f64""",
        "%2",
    )

    assert str(cost_equations(mlir)["fp32"]) == "x*(x + y + 1)"


def test_preserves_dotted_variable_names():
    mlir = category_cost_mlir(
        (
            '%residency.p1: f64 {cost.name = "residency.p1"}, '
            '%residency.l1_cost: f64 {cost.name = "residency.l1_cost"}'
        ),
        "  %0 = arith.mulf %residency.p1, %residency.l1_cost : f64",
        "%0",
    )

    assert (
        str(cost_equations(mlir)["fp32"])
        == "residency.l1_cost*residency.p1"
    )


def test_rejects_unsupported_operation():
    mlir = category_cost_mlir(
        '%x: f64 {cost.name = "x"}',
        "  %0 = arith.divf %x, %x : f64",
        "%0",
    )

    with pytest.raises(ValueError, match="unsupported op"):
        cost_equations(mlir)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="input MLIR is empty"):
        cost_equations("")


def test_cli_reports_validation_errors_without_traceback(tmp_path):
    empty_mlir = tmp_path / "empty.mlir"
    empty_mlir.write_text("")

    result = subprocess.run(
        [sys.executable, "-m", PARSE_MODULE, str(empty_mlir)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "error: input MLIR is empty\n"
    assert "Traceback" not in result.stderr


def test_cli_reads_stdin_and_labels_every_category():
    result = subprocess.run(
        [sys.executable, "-m", PARSE_MODULE, "-"],
        input=SAMPLE_COST_MLIR,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "fp32: 4096.0*dpas_cost + 3505.0",
        "fp64: 0.0",
        "sfu: 0.0",
        "tensor: 0.0",
        "l1: 0.0",
        "memory: 0.0",
    ]
