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
            "usage: verify_float_comparison_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )
    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    symbol = sympy.Symbol
    cmp32 = symbol("arith.cmpf32")
    cmp64 = symbol("arith.cmpf64")
    cmp16 = symbol("arith.cmpf16")
    select = symbol("arith.select")
    store = symbol("triton.store_cost")
    cases = {
        "compare_scalar": {"fp32": 32 * (cmp32 + select), "memory": 128 * store},
        "compare_tensor": {"fp32": 128 * cmp32},
        # x == x still needs a NaN-aware comparison; it is not constant true.
        "compare_self": {"fp32": 128 * cmp32},
        "compare_integer": {"int32": 128 * symbol("arith.cmpi")},
        "compare_f64": {"fp64": 32 * cmp64},
        "compare_tensor_f64": {"fp64": 128 * cmp64},
        "compare_f16": {"fp32": 128 * cmp16},
        "compare_bf16": {"fp32": 128 * cmp16},
        "compare_always_true": {},
        "compare_always_false": {},
    }
    rejected = {
        "compare_f64": "arith.cmpf64",
        "compare_tensor_f64": "arith.cmpf64",
        "compare_f16": "arith.cmpf16",
        "compare_bf16": "arith.cmpf16",
    }

    def check(fixture: Path, function: str, expected: dict) -> None:
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
            assert function in rejected, (function, str(error))
            assert f"unmodeled cost weight {rejected[function]}:" in str(error), str(error)
        else:
            assert function not in rejected, function
            substitutions = {
                symbol(name): config["parameters"][name]["work_per_count"]
                for name in parsed.weight_symbols
            }
            assert work == {
                category: float(expression.subs(substitutions))
                for category, expression in equations.items()
            }

    for function, expected in cases.items():
        check(ttgir, function, expected)

    predicates = (
        "oeq", "one", "ogt", "oge", "olt", "ole", "ord",
        "ueq", "ugt", "uge", "ult", "ule", "une", "uno",
        "true", "false",
    )
    source = ttgir.read_text()
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "predicate.ttgir"
        for predicate in predicates:
            fixture.write_text(source.replace(
                "arith.cmpf ogt,", f"arith.cmpf {predicate},"
            ))
            comparison = 0 if predicate in ("true", "false") else cmp32
            check(fixture, "compare_scalar", {
                "fp32": 32 * (comparison + select), "memory": 128 * store,
            })
            check(fixture, "compare_tensor", {"fp32": 128 * comparison})

            # The public kernel uses dynamic, unrelated select values, keeping
            # the predicate live without introducing min/max folding.
            if predicate not in ("true", "false"):
                lowered = subprocess.run(
                    [str(triton_opt), str(fixture), "--symbol-dce",
                     "--allocate-shared-memory-nv=compute-capability=86 ptx-version=83",
                     "--convert-triton-gpu-to-llvm=compute-capability=86 ptx-version=83",
                     "--convert-nv-gpu-to-llvm"],
                    text=True, capture_output=True, check=True, timeout=10.0,
                ).stdout
                assert f'llvm.fcmp "{predicate}"' in lowered, lowered


if __name__ == "__main__":
    main()
