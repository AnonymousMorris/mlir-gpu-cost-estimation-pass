from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import sympy
from src.cost_model import (
    PIPELINES,
    eval_work,
    grid_program_count,
    per_sm_throughput_rates,
    pipeline_exprs,
    predict_ms,
)
from src.cost_pass import extract_cost_function, pass_pipeline
from src.mlir_sympy import build_equations, parse_cost_function
from src.parse import block_metadata, cost_equations
from src.plotting import kernel_color_map, plot, plottable_rows
from src.results import CostResult, summarize
from src.scheduler import schedule_work

CATEGORY_COST_MLIR = """func.func @__cost_expr(
  %arith.addf32: f64 {cost.name = "arith.addf32"},
  %math.exp: f64 {cost.name = "math.exp"},
  %triton.dot_cost: f64 {cost.name = "triton.dot_cost"},
  %triton.load_cost: f64 {cost.name = "triton.load_cost"},
  %triton_gpu.local_load_cost: f64 {cost.name = "triton_gpu.local_load_cost"}
) -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "int32"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "l1"},
  f64 {cost.name = "memory"}
) {
  %c0 = arith.constant 0.000000e+00 : f64
  %c2 = arith.constant 2.000000e+00 : f64
  %fp32 = arith.mulf %arith.addf32, %c2 : f64
  return %fp32, %c0, %c0, %math.exp, %triton.dot_cost, %triton_gpu.local_load_cost, %triton.load_cost : f64, f64, f64, f64, f64, f64, f64
}"""


BLOCK_COST_MLIR = """func.func @__cost_expr() -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "int32"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "l1"},
  f64 {cost.name = "memory"}
) attributes {
  cost.num_ctas = 2 : i64,
  cost.threads_per_block = 128 : i64,
  cost.work_unit = "block"
} {
  %c0 = arith.constant 0.000000e+00 : f64
  return %c0, %c0, %c0, %c0, %c0, %c0, %c0 : f64, f64, f64, f64, f64, f64, f64
}"""


DIVISION_COST_MLIR = """func.func @__cost_expr() -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "int32"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "l1"},
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
  return %total, %c0, %c0, %c0, %c0, %c0, %c0 : f64, f64, f64, f64, f64, f64, f64
}"""


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]


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
        category_expressions={pipeline: "1.0" for pipeline in PIPELINES},
        args=[],
        kwargs={},
        scalar_args={},
        grid_size=[1],
        schedule=SCHEDULE,
    )


def test_builds_one_equation_for_each_named_cost_result():
    equations = build_equations(parse_cost_function(CATEGORY_COST_MLIR))

    assert tuple(equations) == PIPELINES
    assert equations == {
        "fp32": 2.0 * sympy.Symbol("arith.addf32"),
        "fp64": sympy.Float(0),
        "int32": sympy.Float(0),
        "sfu": sympy.Symbol("math.exp"),
        "tensor": sympy.Symbol("triton.dot_cost"),
        "l1": sympy.Symbol("triton_gpu.local_load_cost"),
        "memory": sympy.Symbol("triton.load_cost"),
    }


def test_public_parser_returns_category_equations():
    equations = cost_equations(CATEGORY_COST_MLIR)

    assert tuple(equations) == PIPELINES
    assert equations["sfu"] == sympy.Symbol("math.exp")
    assert equations["l1"] == sympy.Symbol("triton_gpu.local_load_cost")
    assert equations["memory"] == sympy.Symbol("triton.load_cost")


def test_parser_handles_signed_and_unsigned_integer_division():
    equations = build_equations(parse_cost_function(DIVISION_COST_MLIR))

    assert equations["fp32"] == 10


def test_pipeline_expressions_use_result_categories_without_reclassification():
    expressions = pipeline_exprs(cost_equations(CATEGORY_COST_MLIR))

    assert tuple(expressions) == PIPELINES
    assert expressions["l1"] == sympy.Symbol("triton_gpu.local_load_cost")
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


def test_pipeline_expressions_require_the_complete_category_contract():
    incomplete = """func.func @__cost_expr() -> (f64 {cost.name = "fp32"}) {
      %c0 = arith.constant 0.000000e+00 : f64
      return %c0 : f64
    }"""

    with pytest.raises(ValueError, match="missing categories"):
        pipeline_exprs(cost_equations(incomplete))


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


def test_evaluates_runtime_symbols_separately_from_cost_weights():
    K = sympy.Symbol("K")
    expressions = {pipeline: sympy.Float(0) for pipeline in PIPELINES}
    expressions["tensor"] = 2_097_152 * sympy.ceiling(K / 64)

    work = eval_work(
        expressions,
        {},
        {"K": 256},
        weight_symbols=frozenset(),
        runtime_symbols=frozenset({"K"}),
    )

    assert work["tensor"] == 8_388_608.0


def test_rejects_missing_runtime_symbol_bindings():
    K = sympy.Symbol("K")
    expressions = {pipeline: sympy.Float(0) for pipeline in PIPELINES}
    expressions["tensor"] = K

    with pytest.raises(ValueError, match="missing runtime argument bindings: K"):
        eval_work(
            expressions,
            {},
            {},
            weight_symbols=frozenset(),
            runtime_symbols=frozenset({"K"}),
        )


def test_applies_configured_weight_in_category_units():
    weight = sympy.Symbol("new.operation")
    expressions = {pipeline: sympy.Float(0) for pipeline in PIPELINES}
    expressions["fp32"] = 4 * weight

    work = eval_work(
        expressions,
        {"parameters": {"new.operation": {"work_per_count": 2.0}}},
        {},
        weight_symbols=frozenset({"new.operation"}),
        runtime_symbols=frozenset(),
    )

    assert work["fp32"] == 8.0


@pytest.mark.parametrize("parameter", [{}, {"ops_per_count": 1.0}])
def test_rejects_missing_or_legacy_cost_weights(parameter):
    weight = sympy.Symbol("new.operation")
    with pytest.raises(ValueError, match="missing work_per_count"):
        eval_work(
            {"fp32": weight},
            {"parameters": {"new.operation": parameter}},
            {},
            weight_symbols=frozenset({"new.operation"}),
            runtime_symbols=frozenset(),
        )


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_rejects_invalid_cost_weights(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        eval_work(
            {"int32": sympy.Symbol("arith.addi")},
            {"parameters": {"arith.addi": {"work_per_count": weight}}},
            {},
            weight_symbols=frozenset({"arith.addi"}),
            runtime_symbols=frozenset(),
        )


@pytest.mark.parametrize("name", [
    "arith.ceildivsi", "arith.ceildivui",
    "arith.divf16", "arith.divf32", "arith.divf64",
    "arith.cmpf16", "arith.cmpf64",
    "arith.divsi16", "arith.divsi32", "arith.divsi64", "arith.divui",
    "arith.remsi16", "arith.remsi32", "arith.remsi64", "arith.remui",
    "triton.extern.__nv_asinf",
    "arith.extf.f16_to_f64", "arith.extf.bf16_to_f64", "arith.extf.f32_to_f64",
])
def test_unmodeled_costs_require_explicit_calibration(name):
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    parameter = config["parameters"][name]
    assert parameter["work_per_count"] is None
    assert parameter["unmodeled_reason"]

    with pytest.raises(ValueError, match="unmodeled cost weight") as error:
        eval_work(
            {"fp32": 4 * sympy.Symbol(name)}, config, {},
            weight_symbols=frozenset({name}), runtime_symbols=frozenset(),
        )
    assert name in str(error.value)
    assert parameter["unmodeled_reason"] in str(error.value)

    # Explicit calibration is allowed; this test value is not a hardware claim.
    parameter["work_per_count"] = 8.0
    work = eval_work(
        {"fp32": 4 * sympy.Symbol(name)}, config, {},
        weight_symbols=frozenset({name}), runtime_symbols=frozenset(),
    )
    assert work["fp32"] == 32.0


def test_unused_unmodeled_weight_does_not_reject_a_prediction():
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    assert eval_work(
        {"int32": sympy.Integer(0)}, config, {},
        weight_symbols=frozenset({"arith.divsi64"}), runtime_symbols=frozenset(),
    ) == {"int32": 0.0}


def test_scalar_fma_uses_one_instruction_not_two_flops():
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    for width in (16, 32, 64):
        for op in ("addf", "mulf", "fma"):
            assert config["parameters"][f"arith.{op}{width}"]["work_per_count"] == 1.0


def test_signed_i32_division_uses_explicit_instruction_mixes():
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    for operation, integer_work in (("divsi", 18.0), ("remsi", 16.0)):
        expected = {
            "constant.int32_cost": 1.0,
            "runtime.int32_cost": integer_work,
            "runtime.fp32_cost": 2.0,
            "runtime.sfu_cost": 1.0,
        }
        for suffix, value in expected.items():
            name = f"arith.{operation}32.{suffix}"
            assert config["parameters"][name]["work_per_count"] == value
        assert f"arith.{operation}" not in config["parameters"]


def test_fp32_comparisons_count_one_instruction():
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    assert config["parameters"]["arith.cmpf32"]["work_per_count"] == 1.0
    assert "arith.cmpf" not in config["parameters"]
    assert "arith.cmpf1" not in config["parameters"]


def test_ampere_float_division_uses_an_instruction_mix():
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    expected = {
        "arith.divf32.fp32_cost": 7.0,
        "arith.divf32.sfu_cost": 1.0,
        "arith.reciprocal32.fp32_cost": 6.0,
        "arith.reciprocal32.sfu_cost": 1.0,
    }
    for name, work in expected.items():
        assert config["parameters"][name]["work_per_count"] == work
    assert "arith.divf" not in config["parameters"]


def test_native_float_extensions_count_one_instruction():
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    for name in ("arith.extf.f16_to_f32", "arith.extf.bf16_to_f32"):
        assert config["parameters"][name]["work_per_count"] == 1.0


def test_layout_movement_has_explicit_per_unit_weights():
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    assert "triton_gpu.convert_layout_cost" not in config["parameters"]
    for kind in ("shuffle", "select", "shared_byte", "warp_sync", "cta_barrier"):
        name = f"triton_gpu.layout.{kind}_cost"
        assert config["parameters"][name]["work_per_count"] == 1.0


def test_memory_parameters_do_not_rescale_emitted_byte_counts():
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())

    for name in (
        "triton.load_cost", "triton.store_cost",
        "triton_gpu.local_alloc_cost", "triton_gpu.local_load_cost",
    ):
        assert config["parameters"][name]["work_per_count"] == 1.0


def test_builds_per_sm_throughput_rates():
    spec = {
        "sms": 30,
        "assumed_clock_ghz": 1.0,
        "memory_bandwidth_gb_s": 300.0,
        "per_sm_instructions_per_cycle": {
            "fp32": 128,
            "fp64": 2,
            "int32": 64,
            "fp32_int32": 128,
            "sfu": 16,
        },
        "per_sm_flops_per_cycle": {"tensor_tf32": 512},
        "per_sm_bytes_per_cycle": {"l1": 128},
        "utilization": {},
    }

    rates = per_sm_throughput_rates(spec, {})

    assert rates["fp32"] == 128_000_000.0
    assert rates["fp64"] == 2_000_000.0
    assert rates["int32"] == 64_000_000.0
    assert rates["fp32_int32"] == 128_000_000.0
    assert rates["sfu"] == 16_000_000.0
    assert rates["tensor"] == 512_000_000.0
    assert rates["l1"] == 128_000_000.0
    assert rates["memory"] == 10_000_000.0


@pytest.mark.parametrize(
    ("fp32", "int32", "expected_cycles", "bottleneck"),
    [
        (128, 0, 1.0, "fp32"),
        (0, 64, 1.0, "int32"),
        (64, 64, 1.0, "int32"),
        (128, 64, 1.5, "fp32_int32"),
    ],
)
def test_ga10x_execution_capacity(fp32, int32, expected_cycles, bottleneck):
    spec = json.loads((ANALYSIS_ROOT / "gpu_spec.json").read_text())
    spec["assumed_clock_ghz"] = 1.0
    spec["utilization"] = {}
    rates = per_sm_throughput_rates(spec, {})
    work = dict.fromkeys(PIPELINES, 0.0)
    work.update(fp32=fp32, int32=int32)

    predicted, actual_bottleneck, times = predict_ms(work, rates, 0.0)

    assert predicted == pytest.approx(expected_cycles / 1_000_000)
    assert actual_bottleneck == bottleneck
    assert times["fp32_int32"] == pytest.approx((fp32 + int32) / 128_000_000)


def test_architectures_without_shared_alu_capacity():
    spec = json.loads((ANALYSIS_ROOT / "gpu_spec.json").read_text())
    del spec["per_sm_instructions_per_cycle"]["fp32_int32"]
    rates = per_sm_throughput_rates(spec, {})
    assert "fp32_int32" not in rates
    _, _, times = predict_ms(dict.fromkeys(PIPELINES, 0.0), rates, 0.0)
    assert "fp32_int32" not in times


def test_tensor_capacity_is_in_flops_for_selected_path():
    spec = json.loads((ANALYSIS_ROOT / "gpu_spec.json").read_text())
    spec["assumed_clock_ghz"] = 1.0
    spec["utilization"] = {}
    rates = per_sm_throughput_rates(spec, {"tensor_throughput_key": "tensor_fp16"})
    work = dict.fromkeys(PIPELINES, 0.0)
    work["tensor"] = 1024  # 512 multiply-accumulates.
    predicted, bottleneck, _ = predict_ms(work, rates, 0.0)
    assert predicted == pytest.approx(1 / 1_000_000)
    assert bottleneck == "tensor"


@pytest.mark.parametrize("capacity", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_resource_capacity(capacity):
    spec = json.loads((ANALYSIS_ROOT / "gpu_spec.json").read_text())
    spec["per_sm_instructions_per_cycle"]["int32"] = capacity
    with pytest.raises(ValueError, match="int32"):
        per_sm_throughput_rates(spec, {})


def test_predicts_l1_bottleneck():
    work = {pipeline: 0.0 for pipeline in PIPELINES}
    work["l1"] = 2.0
    rates = {pipeline: 1.0 for pipeline in PIPELINES}

    predicted_ms, bottleneck, pipeline_ms = predict_ms(work, rates, 0.5)

    assert bottleneck == "l1"
    assert pipeline_ms["l1"] == 2.0
    assert predicted_ms == 2.5


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


def test_assigns_stable_distinct_colors_to_kernels():
    rows = [result("softmax_kernel"), result("add_kernel")]

    colors = kernel_color_map(rows)

    assert list(colors) == ["add_kernel", "softmax_kernel"]
    assert len(set(colors.values())) == 2


@pytest.mark.parametrize("bottleneck", ["memory", "int32", "fp32_int32"])
def test_plot_creates_nested_plot_directory(tmp_path, bottleneck):
    plots_dir = tmp_path / "nested" / "plots"
    row = replace(result("add_kernel", predicted_ms=0.02), bottleneck=bottleneck)

    plot(
        [row],
        plots_dir / "scatter.png",
        plots_dir / "pipelines.png",
    )

    assert (plots_dir / "scatter.png").is_file()
    assert (plots_dir / "pipelines.png").is_file()
