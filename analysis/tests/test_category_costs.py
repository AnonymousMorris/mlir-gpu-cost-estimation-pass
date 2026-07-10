from __future__ import annotations

import sympy
import pytest

from analyze_costs import (
    CostResult,
    PIPELINES,
    extract_cost_function,
    grid_program_count,
    launch_rates,
    launch_work,
    pipeline_exprs,
    plot,
    plottable_rows,
    program_metadata,
    summarize,
)
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


PROGRAM_COST_MLIR = """func.func @__cost_expr() -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "memory"}
) attributes {
  cost.num_ctas = 2 : i64,
  cost.threads_per_program = 256 : i64,
  cost.work_unit = "program"
} {
  %c0 = arith.constant 0.000000e+00 : f64
  return %c0, %c0, %c0, %c0, %c0 : f64, f64, f64, f64, f64
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


def test_extracts_function_with_execution_metadata_attributes():
    extracted = extract_cost_function("pass output\n" + PROGRAM_COST_MLIR + "\nmore output")

    assert extracted == PROGRAM_COST_MLIR


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


def test_reads_program_execution_metadata():
    assert program_metadata(PROGRAM_COST_MLIR).num_ctas == 2
    assert program_metadata(PROGRAM_COST_MLIR).threads_per_program == 256


def test_grid_program_count_multiplies_all_dimensions():
    assert grid_program_count([2, 3, 4]) == 24


@pytest.mark.parametrize("grid_size", [[], [0], [4, -1]])
def test_grid_program_count_rejects_invalid_grids(grid_size):
    with pytest.raises(ValueError, match="grid_size"):
        grid_program_count(grid_size)


def test_launch_work_scales_per_program_work_by_grid_volume():
    per_program = {pipeline: float(index) for index, pipeline in enumerate(PIPELINES)}

    assert launch_work(per_program, [2, 3, 4]) == {
        pipeline: value * 24 for pipeline, value in per_program.items()
    }


def test_launch_rates_account_for_small_grid_sm_underfill():
    rates = {pipeline: 300.0 for pipeline in PIPELINES}

    assert launch_rates(rates, sms=30, programs=4, num_ctas=1) == {
        pipeline: 40.0 for pipeline in PIPELINES
    }
    assert launch_rates(rates, sms=30, programs=60, num_ctas=1) == rates


def test_summary_reports_symmetric_multiplicative_accuracy():
    rows = [
        CostResult(
            kernel=f"kernel_{index}",
            ttgir=f"kernel_{index}.ttgir",
            time_ms=1.0,
            predicted_ms=predicted_ms,
            bottleneck="memory",
            pipeline_work={pipeline: 1.0 for pipeline in PIPELINES},
            pipeline_ms={pipeline: 1.0 for pipeline in PIPELINES},
            expression="1.0",
            category_expressions={pipeline: "1.0" for pipeline in PIPELINES},
            args=[],
            kwargs={},
            grid_size=[1],
        )
        for index, predicted_ms in enumerate((0.25, 0.5, 1.0, 2.0, 10.0))
    ]

    summary = summarize(rows, [])

    assert summary["median_multiplicative_error"] == 2.0
    assert summary["within_2x_fraction"] == 0.6
    assert summary["within_5x_fraction"] == 0.8


def test_plots_exclude_persistent_matmul_rows():
    def row(kernel):
        return CostResult(
            kernel=kernel,
            ttgir=f"{kernel}.ttgir",
            time_ms=1.0,
            predicted_ms=1.0,
            bottleneck="memory",
            pipeline_work={pipeline: 1.0 for pipeline in PIPELINES},
            pipeline_ms={pipeline: 1.0 for pipeline in PIPELINES},
            expression="1.0",
            category_expressions={pipeline: "1.0" for pipeline in PIPELINES},
            args=[],
            kwargs={},
            grid_size=[1],
        )

    regular = row("matmul_kernel")
    persistent = row("matmul_kernel_persistent")

    assert plottable_rows([regular, persistent]) == [regular]


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
