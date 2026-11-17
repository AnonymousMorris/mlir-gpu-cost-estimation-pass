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
            "usage: verify_integer_division_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )
    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    symbol = sympy.Symbol
    div_const = symbol("arith.divsi32.constant.int32_cost")
    rem_const = symbol("arith.remsi32.constant.int32_cost")
    store = symbol("triton.store_cost")

    def runtime(operation: str, count: int) -> dict:
        return {
            category: count * symbol(f"arith.{operation}32.runtime.{category}_cost")
            for category in ("int32", "fp32", "sfu")
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

    cases = {
        "divide_runtime": runtime("divsi", 32) | {"memory": 128 * store},
        "remainder_runtime": runtime("remsi", 32) | {"memory": 128 * store},
        "divide_tensor": runtime("divsi", 128),
        "divide_dynamic_splat": runtime("divsi", 128) | {
            "fp32": 128 * (symbol("arith.divsi32.runtime.fp32_cost")
                           + symbol("triton.splat_cost")),
        },
        "remainder_tensor": runtime("remsi", 128),
        "divide_constant": {"int32": 96 * div_const},
        "remainder_constant": {"int32": 128 * rem_const},
        "divide_dense": {"int32": 384 * div_const},
        "remainder_splat": {
            "int32": 512 * rem_const, "fp32": 128 * symbol("triton.splat_cost"),
        },
        "fold_constants": {},
        "remainder_min_negative_one": {},
        # The layout has two register elements per lane, including replication.
        "mixed_divisors": runtime("divsi", 64),
    }
    for function, expected in cases.items():
        check(ttgir, function, expected)
    for function, weight in (
        ("divide_i16", "arith.divsi16"), ("remainder_i16", "arith.remsi16"),
        ("divide_i64", "arith.divsi64"), ("remainder_i64", "arith.remsi64"),
        ("unsigned_divide", "arith.divui"), ("unsigned_remainder", "arith.remui"),
    ):
        check(ttgir, function, {"int32": 32 * symbol(weight)}, weight)

    try:
        run_cost_pass(triton_opt, plugin, ttgir, "divide_index", timeout_s=10.0)
    except RuntimeError as error:
        assert "explicit integer or floating-point bit width" in str(error), str(error)
    else:
        raise AssertionError("index division silently acquired an i32 cost")

    # Expected fused Ampere arithmetic counts, excluding register moves.
    # In particular, signed powers of two require bias, not just one shift.
    constant_counts = {
        1: (0, 0), -1: (1, 0),
        2: (2, 3), -2: (3, 3), 8: (3, 4), -8: (4, 4),
        3: (2, 3), -3: (4, 3), 5: (3, 4), -5: (3, 4),
        7: (3, 4), -7: (4, 4), 48: (3, 4), 127: (3, 4),
        2147483647: (3, 4), -2147483648: (2, 4),
    }
    source = ttgir.read_text()
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "case.ttgir"
        for divisor, (div_count, rem_count) in constant_counts.items():
            fixture.write_text(source.replace(
                "arith.constant 8 : i32", f"arith.constant {divisor} : i32"
            ))
            check(fixture, "divide_constant", {"int32": 32 * div_count * div_const})
            check(fixture, "remainder_constant", {"int32": 32 * rem_count * rem_const})

        for target in ("80", "87", "90"):
            fixture.write_text(source.replace("cuda:86", f"cuda:{target}"))
            for function, op in (("divide_tensor", "divsi"), ("remainder_tensor", "remsi")):
                if target == "90":
                    weight = f"arith.{op}32"
                    check(fixture, function, {"int32": 128 * symbol(weight)}, weight)
                else:
                    check(fixture, function, cases[function])

        # Invalid constant operations must fail, not become free constant folds.
        header = source.split("  tt.func", 1)[0]
        for op in ("arith.divsi", "arith.remsi"):
            invalid = [
                ("%x: i32", f"%d = arith.constant 0 : i32\n"
                 f"    %r = {op} %x, %d : i32", "by zero"),
                ("%x: tensor<4xi32, #blocked>",
                 "%d = arith.constant dense<[2, 0, 4, 5]> : tensor<4xi32, #blocked>\n"
                 f"    %r = {op} %x, %d : tensor<4xi32, #blocked>", "by zero"),
            ]
            if op == "arith.divsi":
                invalid += [
                    ("", "%x = arith.constant -2147483648 : i32\n"
                     "    %d = arith.constant -1 : i32\n"
                     f"    %r = {op} %x, %d : i32", "INT_MIN"),
                    ("", "%x = arith.constant dense<[-2147483648, 0, 1, 2]> : tensor<4xi32, #blocked>\n"
                     "    %d = arith.constant dense<-1> : tensor<4xi32, #blocked>\n"
                     f"    %r = {op} %x, %d : tensor<4xi32, #blocked>", "INT_MIN"),
                    ("", "%x = arith.constant dense<[-2147483648, 0, 1, 2]> : tensor<4xi32, #blocked>\n"
                     "    %d = arith.constant dense<[-1, 2, 3, 4]> : tensor<4xi32, #blocked>\n"
                     f"    %r = {op} %x, %d : tensor<4xi32, #blocked>", "INT_MIN"),
                ]
            for args, operations, diagnostic in invalid:
                fixture.write_text(
                    header + f"  tt.func public @invalid({args}) {{\n    "
                    + operations + "\n    tt.return\n  }\n}\n"
                )
                try:
                    run_cost_pass(triton_opt, plugin, fixture, "invalid", timeout_s=10.0)
                except RuntimeError as error:
                    assert diagnostic in str(error), str(error)
                else:
                    raise AssertionError(f"accepted invalid {op}")

        # INT_MIN and -1 in different lanes must not be mistaken for overflow.
        fixture.write_text(
            header + "  tt.func public @valid() {\n"
            "    %x = arith.constant dense<[-2147483648, 0, 1, 2]> : tensor<4xi32, #blocked>\n"
            "    %d = arith.constant dense<[2, -1, 3, 4]> : tensor<4xi32, #blocked>\n"
            "    %r = arith.divsi %x, %d : tensor<4xi32, #blocked>\n"
            "    tt.return\n  }\n}\n"
        )
        check(fixture, "valid", runtime("divsi", 64))

    lowered = subprocess.run(
        [str(triton_opt), str(ttgir), "--symbol-dce",
         "--allocate-shared-memory-nv=compute-capability=86 ptx-version=83",
         "--convert-triton-gpu-to-llvm=compute-capability=86 ptx-version=83",
         "--convert-nv-gpu-to-llvm"],
        text=True, capture_output=True, check=True, timeout=10.0,
    ).stdout
    assert lowered.count("llvm.sdiv ") == 1, lowered
    assert lowered.count("llvm.srem ") == 1, lowered


if __name__ == "__main__":
    main()
