from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import sympy

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.cost_model import eval_work, pipeline_exprs
from src.cost_pass import run_cost_pass
from src.parse import parse_cost_analysis


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_views_extensions_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )
    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    symbol = sympy.Symbol
    f16 = symbol("arith.extf.f16_to_f32")
    bf16 = symbol("arith.extf.bf16_to_f32")
    store = symbol("triton.store_cost")
    layout = symbol("triton_gpu.convert_layout_cost")
    expected = {
        "bitcast_scalar": {},
        "bitcast_tensor": {},
        "bitcast_then_add": {"fp32": 128 * symbol("arith.addf32")},
        "transpose": {},
        # The existing layout proxy uses max(src, dst) elements per thread.
        "transpose_then_convert": {"fp32": 256 * layout},
        "extend_scalar_f16": {"fp32": 32 * f16, "memory": 128 * store},
        "extend_scalar_bf16": {"int32": 32 * bf16, "memory": 128 * store},
        "extend_tensor_f16": {"fp32": 128 * f16},
        "extend_tensor_bf16": {"int32": 128 * bf16},
        "extend_f32_f64": {"fp32": 128 * symbol("arith.extf.f32_to_f64")},
        "extend_unconfigured": {"fp32": 32 * symbol("arith.extf.f64_to_f128")},
    }
    rejected = {
        "transpose_then_convert": "unmodeled cost weight triton_gpu.convert_layout_cost",
        "extend_f32_f64": "unmodeled cost weight arith.extf.f32_to_f64",
        "extend_unconfigured": "missing work_per_count for cost weight: arith.extf.f64_to_f128",
    }
    for function, costs in expected.items():
        parsed = parse_cost_analysis(run_cost_pass(
            triton_opt, plugin, ttgir, function, timeout_s=10.0
        ))
        equations = pipeline_exprs(parsed.equations)
        for category, expression in equations.items():
            difference = sympy.simplify(expression - costs.get(category, 0))
            assert difference.is_zero, (function, category, expression)
        try:
            work = eval_work(
                equations, config, {},
                weight_symbols=parsed.weight_symbols,
                runtime_symbols=parsed.runtime_symbols,
            )
        except ValueError as error:
            assert function in rejected, (function, str(error))
            assert rejected[function] in str(error), str(error)
        else:
            assert function not in rejected, function
            assert work == {
                category: float(sympy.sympify(costs.get(category, 0)).subs({
                    symbol(weight): 1 for weight in parsed.weight_symbols
                }))
                for category in equations
            }, (function, work)

    # Invalid reinterpretations/layouts must fail verification, not acquire a
    # zero cost merely because the operation name is recognized.
    source = ttgir.read_text()
    invalid = [
        ("bitcast_scalar", source.replace(
            "%y = tt.bitcast %x : i32 -> f32",
            "%y = tt.bitcast %x : i32 -> f64",
        ), "Cannot bitcast data-type"),
        ("bitcast_scalar", source.replace(
            "%p = tt.bitcast %ptr : !tt.ptr<f16> -> !tt.ptr<f32>",
            "%p = tt.bitcast %ptr : !tt.ptr<f16> -> i64",
        ), "Cannot bitcast pointer to non-pointer type"),
        ("transpose", source.replace(
            "tensor<8x16xf32, #matrix> -> tensor<16x8xf32, #transposed>",
            "tensor<8x16xf32, #matrix> -> tensor<16x8xf32, #matrix>",
        ), "tt.trans"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "invalid.ttgir"
        for function, text, diagnostic in invalid:
            path.write_text(text)
            try:
                run_cost_pass(triton_opt, plugin, path, function, timeout_s=10.0)
            except RuntimeError as error:
                assert diagnostic in str(error), str(error)
            else:
                raise AssertionError(f"accepted invalid {function}")

    # Private cost-only fixtures are discarded. The two public kernels store
    # their conversion results, keeping the native fpext operations live.
    lowered = subprocess.run(
        [str(triton_opt), str(ttgir), "--symbol-dce",
         "--allocate-shared-memory-nv=compute-capability=86 ptx-version=83",
         "--convert-triton-gpu-to-llvm=compute-capability=86 ptx-version=83",
         "--convert-nv-gpu-to-llvm"],
        text=True, capture_output=True, check=True, timeout=10.0,
    ).stdout
    assert lowered.count("llvm.fpext") == 2, lowered
    assert ": f16 to f32" in lowered
    assert ": bf16 to f32" in lowered


if __name__ == "__main__":
    main()
