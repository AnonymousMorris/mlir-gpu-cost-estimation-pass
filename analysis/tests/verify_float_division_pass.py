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
            "usage: verify_float_division_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )
    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    symbol = sympy.Symbol
    divide_fp32 = symbol("arith.divf32.fp32_cost")
    divide_sfu = symbol("arith.divf32.sfu_cost")
    reciprocal_fp32 = symbol("arith.reciprocal32.fp32_cost")
    reciprocal_sfu = symbol("arith.reciprocal32.sfu_cost")
    store = symbol("triton.store_cost")
    cases = {
        "divide_scalar": {
            "fp32": 32 * divide_fp32, "sfu": 32 * divide_sfu,
            "memory": 128 * store,
        },
        "reciprocal_scalar": {
            "fp32": 32 * reciprocal_fp32, "sfu": 32 * reciprocal_sfu,
            "memory": 128 * store,
        },
        "divide_tensor": {"fp32": 128 * divide_fp32, "sfu": 128 * divide_sfu},
        "reciprocal_tensor": {
            "fp32": 128 * reciprocal_fp32, "sfu": 128 * reciprocal_sfu,
        },
        "reciprocal_splat": {
            "fp32": 128 * (reciprocal_fp32 + symbol("triton.splat_cost")),
            "sfu": 128 * reciprocal_sfu,
        },
        "nonunit_numerator": {"fp32": 128 * divide_fp32, "sfu": 128 * divide_sfu},
        "divide_f64": {"fp64": 32 * symbol("arith.divf64")},
        "divide_tensor_f64": {"fp64": 128 * symbol("arith.divf64")},
        "divide_f16": {"fp32": 32 * symbol("arith.divf16")},
    }

    def check(
        fixture: Path, function: str, expected: dict, rejected: str = ""
    ) -> None:
        parsed = parse_cost_analysis(run_cost_pass(
            triton_opt, plugin, fixture, function, timeout_s=10.0
        ))
        equations = pipeline_exprs(parsed.equations)
        for category, expression in equations.items():
            assert sympy.simplify(expression - expected.get(category, 0)).is_zero, (
                function, category, expression
            )
        try:
            work = eval_work(
                equations, config, {},
                weight_symbols=parsed.weight_symbols,
                runtime_symbols=parsed.runtime_symbols,
            )
        except ValueError as error:
            assert rejected, (function, str(error))
            assert f"unmodeled cost weight {rejected}:" in str(error), str(error)
        else:
            assert not rejected, function
            substitutions = {
                symbol(name): config["parameters"][name]["work_per_count"]
                for name in parsed.weight_symbols
            }
            assert work == {
                category: float(expression.subs(substitutions))
                for category, expression in equations.items()
            }

    for function, expected in cases.items():
        rejected = (
            "arith.divf64" if "f64" in function
            else "arith.divf16" if "f16" in function
            else ""
        )
        check(ttgir, function, expected, rejected)

    # Ampere variants use the same mix. Other architectures must not silently
    # reuse those weights, even when their operand type is also FP32.
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "target.ttgir"
        for target in ("80", "87", "90"):
            fixture.write_text(ttgir.read_text().replace("cuda:86", f"cuda:{target}"))
            if target == "90":
                check(fixture, "divide_tensor",
                      {"fp32": 128 * symbol("arith.divf32")}, "arith.divf32")
            else:
                check(fixture, "divide_tensor", cases["divide_tensor"])
                check(fixture, "reciprocal_tensor", cases["reciprocal_tensor"])

    # Only the two public kernels survive DCE; their stores keep results live.
    # In particular, 1/x still uses div.full, not an unchecked rcp.approx.
    lowered = subprocess.run(
        [str(triton_opt), str(ttgir), "--symbol-dce",
         "--allocate-shared-memory-nv=compute-capability=86 ptx-version=83",
         "--convert-triton-gpu-to-llvm=compute-capability=86 ptx-version=83",
         "--convert-nv-gpu-to-llvm"],
        text=True, capture_output=True, check=True, timeout=10.0,
    ).stdout
    assert lowered.count('llvm.call_intrinsic "llvm.nvvm.div.full"') == 2, lowered


if __name__ == "__main__":
    main()
