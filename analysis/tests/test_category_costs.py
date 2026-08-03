from __future__ import annotations

import pytest
import sympy

from analyze_costs import (
    CostResult,
    PIPELINES,
    block_metadata,
    extract_cost_function,
    grid_program_count,
    pass_pipeline,
    per_sm_throughput_rates,
    pipeline_exprs,
    plot,
    plottable_rows,
    summarize,
)
from mlir_sympy import build_equations, parse_cost_function
from parse import cost_equations, formatted_cost_equations
from scheduler import schedule_work


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


BLOCK_COST_MLIR = """func.func @__cost_expr() -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "memory"}
) attributes {
  cost.num_ctas = 2 : i64,
  cost.threads_per_block = 128 : i64,
  cost.work_unit = "block"
} {
  %c0 = arith.constant 0.000000e+00 : f64
  return %c0, %c0, %c0, %c0, %c0 : f64, f64, f64, f64, f64
}"""


DIVISION_COST_MLIR = """func.func @__cost_expr() -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "memory"}
) {
  %c0 = arith.constant 0.000000e+00 : f64
  %c3 = arith.constant 3 : i32
  %c7 = arith.constant 7 : i32
  %divs = arith.divsi %c7, %c3 : i32
  %divu = arith.divui %c7, %c3 : i32
  %ceils = arith.ceildivsi %c7, %c3 : i32
  %ceilu = arith.ceildivui %c7, %c3 : i32
  %divs_f = arith.uitofp %divs : i32 to f64
  %divu_f = arith.uitofp %divu : i32 to f64
  %ceils_f = arith.uitofp %ceils : i32 to f64
  %ceilu_f = arith.uitofp %ceilu : i32 to f64
  %div = arith.addf %divs_f, %divu_f : f64
  %ceil = arith.addf %ceils_f, %ceilu_f : f64
  %total = arith.addf %div, %ceil : f64
  return %total, %c0, %c0, %c0, %c0 : f64, f64, f64, f64, f64
}"""


_, SCHEDULE = schedule_work(
    {pipeline: 1.0 for pipeline in PIPELINES},
    program_count=1,
    num_ctas=1,
    num_sms=30,
)


def result(kernel: str, predicted_ms: float = 1.0) -> CostResult:
    return CostResult(
        kernel=kernel,
        ttgir=f"{kernel}.ttgir",
        time_ms=1.0,
        predicted_ms=predicted_ms,
        bottleneck="memory",
        scheduled_work={pipeline: 1.0 for pipeline in PIPELINES},
        pipeline_ms={pipeline: 1.0 for pipeline in PIPELINES},
        expression="1.0",
        category_expressions={pipeline: "1.0" for pipeline in PIPELINES},
        args=[],
        kwargs={},
        grid_size=[1],
        schedule=SCHEDULE,
    )


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


def test_parser_handles_signed_and_unsigned_integer_division():
    equations = build_equations(parse_cost_function(DIVISION_COST_MLIR))

    assert equations["fp32"] == 10


def test_pipeline_expressions_use_result_categories_without_reclassification():
    expressions = pipeline_exprs(CATEGORY_COST_MLIR)

    assert tuple(expressions) == PIPELINES
    assert expressions["memory"] == sympy.Symbol("triton.load_cost")
    assert expressions["tensor"] == sympy.Symbol("triton.dot_cost")


def test_extracts_function_with_named_multi_result_attributes():
    extracted = extract_cost_function(
        "pass output\n" + CATEGORY_COST_MLIR + "\nmore output"
    )

    assert extracted == CATEGORY_COST_MLIR


def test_extracts_function_with_execution_metadata_attributes():
    extracted = extract_cost_function(
        "pass output\n" + BLOCK_COST_MLIR + "\nmore output"
    )

    assert extracted == BLOCK_COST_MLIR


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


def test_reads_per_block_metadata():
    metadata = block_metadata(BLOCK_COST_MLIR)

    assert metadata.num_ctas == 2
    assert metadata.threads_per_block == 128


def test_builds_grid_independent_pass_pipeline():
    assert pass_pipeline("kernel") == (
        "builtin.module(my-cost-analysis{func-name=kernel})"
    )


def test_grid_program_count_multiplies_all_dimensions():
    assert grid_program_count([2, 3, 4]) == 24


@pytest.mark.parametrize("grid_size", [[], [0], [4, -1]])
def test_grid_program_count_rejects_invalid_grids(grid_size):
    with pytest.raises(ValueError, match="grid_size"):
        grid_program_count(grid_size)


def test_builds_per_sm_throughput_rates():
    spec = {
        "sms": 30,
        "assumed_clock_ghz": 1.0,
        "memory_bandwidth_gb_s": 300.0,
        "per_sm_ops_per_cycle": {
            "fp32": 128,
            "fp64": 2,
            "sfu": 16,
            "tensor_tf32": 512,
        },
        "utilization": {},
    }

    rates = per_sm_throughput_rates(spec, {})

    assert rates["fp32"] == 128_000_000.0
    assert rates["fp64"] == 2_000_000.0
    assert rates["sfu"] == 16_000_000.0
    assert rates["tensor"] == 512_000_000.0
    assert rates["memory"] == 10_000_000.0


def test_summary_reports_symmetric_multiplicative_accuracy():
    rows = [
        result(f"kernel_{index}", predicted_ms)
        for index, predicted_ms in enumerate((0.25, 0.5, 1.0, 2.0, 10.0))
    ]

    summary = summarize(rows, [])

    assert summary["median_multiplicative_error"] == 2.0
    assert summary["within_2x_fraction"] == 0.6
    assert summary["within_5x_fraction"] == 0.8


def test_plots_exclude_persistent_matmul_rows():
    regular = result("matmul_kernel")
    persistent = result("matmul_kernel_persistent")

    assert plottable_rows([regular, persistent]) == [regular]


def test_plot_creates_nested_plot_directory(tmp_path):
    plots_dir = tmp_path / "nested" / "plots"

    plot(
        [result("add_kernel", predicted_ms=0.02)],
        plots_dir / "scatter.png",
        plots_dir / "pipelines.png",
    )

    assert (plots_dir / "scatter.png").is_file()
    assert (plots_dir / "pipelines.png").is_file()
