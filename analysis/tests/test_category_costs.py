from __future__ import annotations

import sympy
import pytest

from analyze_costs import CostResult, PIPELINES, extract_cost_function, pipeline_exprs, plot
from mlir_sympy import build_equations, parse_cost_function
from parse import cost_equations, formatted_cost_equations


CATEGORY_COST_MLIR = """func.func @__cost_expr(
  %arith.addf32: f64 {cost.name = "arith.addf32"},
  %math.exp: f64 {cost.name = "math.exp"},
  %triton.dot_cost: f64 {cost.name = "triton.dot_cost"},
  %triton.load_cost: f64 {cost.name = "triton.load_cost"}
) -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "memory"}
) {
  %c0 = arith.constant 0.000000e+00 : f64
  %c2 = arith.constant 2.000000e+00 : f64
  %fp32 = arith.mulf %arith.addf32, %c2 : f64
  return %fp32, %c0, %math.exp, %triton.dot_cost, %triton.load_cost : f64, f64, f64, f64, f64
}"""


def test_builds_one_equation_for_each_named_cost_result():
    equations = build_equations(parse_cost_function(CATEGORY_COST_MLIR))

    assert tuple(equations) == PIPELINES
    assert equations == {
        "fp32": 2.0 * sympy.Symbol("arith.addf32"),
        "fp64": sympy.Float(0),
        "sfu": sympy.Symbol("math.exp"),
        "tensor": sympy.Symbol("triton.dot_cost"),
        "memory": sympy.Symbol("triton.load_cost"),
    }


def test_public_parser_returns_category_equations():
    equations = cost_equations(CATEGORY_COST_MLIR)

    assert tuple(equations) == PIPELINES
    assert equations["sfu"] == sympy.Symbol("math.exp")
    assert equations["memory"] == sympy.Symbol("triton.load_cost")


def test_pipeline_expressions_use_result_categories_without_reclassification():
    expressions = pipeline_exprs(CATEGORY_COST_MLIR)

    assert tuple(expressions) == PIPELINES
    assert expressions["memory"] == sympy.Symbol("triton.load_cost")
    assert expressions["tensor"] == sympy.Symbol("triton.dot_cost")


def test_extracts_function_with_named_multi_result_attributes():
    extracted = extract_cost_function("pass output\n" + CATEGORY_COST_MLIR + "\nmore output")

    assert extracted == CATEGORY_COST_MLIR


def test_formats_each_category_as_a_separate_equation():
    output = formatted_cost_equations(CATEGORY_COST_MLIR)

    assert output.splitlines() == [
        "fp32: 2.0*arith.addf32",
        "fp64: 0.0",
        "sfu: math.exp",
        "tensor: triton.dot_cost",
        "memory: triton.load_cost",
    ]


def test_pipeline_expressions_require_the_complete_category_contract():
    incomplete = """func.func @__cost_expr() -> (f64 {cost.name = "fp32"}) {
      %c0 = arith.constant 0.000000e+00 : f64
      return %c0 : f64
    }"""

    with pytest.raises(ValueError, match="missing categories"):
        pipeline_exprs(incomplete)


def test_plot_creates_nested_plot_directory(tmp_path):
    row = CostResult(
        kernel="add_kernel",
        ttgir="add.ttgir",
        time_ms=0.01,
        predicted_ms=0.02,
        bottleneck="memory",
        pipeline_work={pipeline: 1.0 for pipeline in PIPELINES},
        pipeline_ms={pipeline: 0.01 for pipeline in PIPELINES},
        expression="1.0",
        category_expressions={pipeline: "1.0" for pipeline in PIPELINES},
        args=[],
        kwargs={},
        grid_size=[1],
    )
    plots_dir = tmp_path / "nested" / "plots"

    plot([row], plots_dir / "scatter.png", plots_dir / "pipelines.png")

    assert (plots_dir / "scatter.png").is_file()
    assert (plots_dir / "pipelines.png").is_file()
