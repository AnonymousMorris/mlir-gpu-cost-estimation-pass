from __future__ import annotations

import sys
from pathlib import Path

import sympy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation"))

from src.cost_pass import run_cost_pass
from src.parse import cost_equations
from cost_mlir import parse_cost_function as validation_parse
from equation import build_equations as validation_equations


def main() -> None:
    triton_opt, plugin, fixture = map(Path, sys.argv[1:])
    cases = [
        (5, 3, 1, 0),
        (5, 5, 2, 0),
        (5, 6, 2, 1),
        (5, 12, 3, 3),
        (-5, -2, 2, 2),
        (-5, 3, 3, 3),
        (-(2**31), 2**31 - 2, 2, 2**31 - 1),
    ]
    for name in ("signed_i32", "signed_i64", "index_loop", "unsigned_i32",
                 "constant_empty"):
        mlir = run_cost_pass(triton_opt, plugin, fixture, name, timeout_s=10.0)
        parsers = (cost_equations(mlir), validation_equations(validation_parse(mlir)))
        current_cases = cases
        if name == "unsigned_i32":
            current_cases = cases[:4] + [
                (2**31, 2**31 + 3, 2, 2),
                (0, -1, 1, 2**32 - 1),
                (-1, 0, 1, 0),
            ]
        elif name in ("signed_i64", "index_loop"):
            # The subtraction must not overflow, even across the i64 sign bit.
            current_cases = cases + [(-(2**63), 2**63 - 2, 2, 2**63 - 1)]
        elif name == "constant_empty":
            current_cases = [(0, 0, 1, 0)]

        for equations in parsers:
            for lower, upper, step, trips in current_cases:
                substitutions = {
                    sympy.Symbol("lower"): lower,
                    sympy.Symbol("upper"): upper,
                    sympy.Symbol("step"): step,
                    sympy.Symbol("arith.addi"): 1,
                }
                for category, expr in equations.items():
                    actual = expr.subs(substitutions)
                    if category == "int32":
                        for trip_expr in expr.atoms(sympy.ceiling):
                            assert trip_expr.subs(substitutions) == trips
                    expected = sympy.Float(float(32 * trips)) if category == "int32" else 0
                    assert sympy.simplify(actual - expected) == 0, (
                        name, category, lower, upper, step, actual, expected
                    )


if __name__ == "__main__":
    main()
