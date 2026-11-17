from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import sympy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation"))

from src.cost_model import WORK_CATEGORIES, eval_work, per_sm_throughput_rates, predict_ms
from src.cost_pass import run_cost_pass
from src.parse import parse_cost_analysis
from cost_mlir import parse_cost_function as validation_parse
from equation import build_equations as validation_equations


def main() -> None:
    triton_opt, plugin, fixture = map(Path, sys.argv[1:])
    config = json.loads((ROOT / "analysis/cost_analysis_config.json").read_text())
    spec = json.loads((ROOT / "analysis/gpu_spec.json").read_text())
    spec["assumed_clock_ghz"] = 1.0
    spec["utilization"] = {}
    rates = per_sm_throughput_rates(spec)
    # Each dot computes 16 x 8 outputs over K=32, counting two FLOPs/MAC.
    flops = 2 * 16 * 8 * 32
    weight = sympy.Symbol("triton.dot_cost")
    for name, fp16, tf32 in (("fp16", flops, 0), ("bf16", flops, 0),
                             ("tf32", 0, flops), ("mixed", flops, flops)):
        mlir = run_cost_pass(triton_opt, plugin, fixture, name, timeout_s=10.0)
        parsed = parse_cost_analysis(mlir)
        independent = validation_equations(validation_parse(mlir))
        for equations in (parsed.equations, independent):
            assert tuple(equations) == WORK_CATEGORIES
            for category in WORK_CATEGORIES:
                expected = {"tensor_fp16": fp16, "tensor_tf32": tf32}.get(category, 0)
                assert sympy.simplify(equations[category] - expected * weight) == 0
        work = eval_work(parsed.equations, config, {},
                         weight_symbols=parsed.weight_symbols,
                         runtime_symbols=parsed.runtime_symbols)
        predicted, bottleneck, _ = predict_ms(work, rates, 0.0)
        expected_ms = (fp16 / 1024 + tf32 / 512) / 1_000_000
        assert abs(predicted - expected_ms) < 1e-15
        assert bottleneck == "tensor"

    with tempfile.TemporaryDirectory() as directory:
        variant = Path(directory) / "dot.ttgir"
        text = fixture.read_text()
        # inputPrecision is irrelevant for half-precision operands.
        variant.write_text(text.replace(
            "inputPrecision = tf32 : tensor<16x32xf16",
            "inputPrecision = ieee : tensor<16x32xf16",
        ))
        parsed = parse_cost_analysis(run_cost_pass(
            triton_opt, plugin, variant, "fp16", timeout_s=10.0
        ))
        assert sympy.simplify(parsed.equations["tensor_fp16"] - flops * weight) == 0
        # Neither software/IEEE dots nor multi-pass TF32 may inherit a unit
        # single-pass TF32 rate without their own lowering-specific model.
        for name, contents, reason in (
            ("simt_ieee", text, "requires an Ampere MMA dot encoding"),
            ("int8", text, "supports only f16/bf16 and f32 TF32 dots"),
            ("fp16", text.replace("32x8xf16, #b16", "32x8xbf16, #b16"),
             "requires matching dot operand types"),
            # This Triton revision rejects tf32x3 at the MMAv2 IR verifier.
            ("tf32", text.replace("inputPrecision = tf32", "inputPrecision = tf32x3"),
             "unsupported MMA version"),
        ):
            variant.write_text(contents)
            try:
                run_cost_pass(triton_opt, plugin, variant, name, timeout_s=10.0)
            except RuntimeError as error:
                assert reason in str(error)
            else:
                raise AssertionError(f"unsupported {name} dot was accepted")


if __name__ == "__main__":
    main()
