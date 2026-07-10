from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy

from mlir_sympy import build_equation, build_equations, parse_cost_function


def cost_equation(mlir: str) -> sympy.Expr:
    """Parse a legacy cost function that returns one scalar expression."""
    return build_equation(parse_cost_function(mlir))


def cost_equations(mlir: str) -> dict[str, sympy.Expr]:
    """Parse all named category expressions returned by a cost function."""
    return build_equations(parse_cost_function(mlir))


def formatted_cost_equations(mlir: str) -> str:
    equations = cost_equations(mlir)
    if len(equations) == 1:
        return str(next(iter(equations.values())))
    return "\n".join(f"{name}: {expression}" for name, expression in equations.items())


def read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplify generated MLIR cost expressions")
    parser.add_argument("input", help="MLIR file, or '-' for stdin")
    args = parser.parse_args()

    try:
        print(formatted_cost_equations(read_input(args.input)))
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
