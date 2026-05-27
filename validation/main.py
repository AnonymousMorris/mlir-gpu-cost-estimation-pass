from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy

from cost_mlir import parse_cost_function
from equation import build_equation


def cost_equation(mlir: str) -> sympy.Expr:
    return build_equation(parse_cost_function(mlir))


def read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplify generated MLIR cost expressions")
    parser.add_argument("input", help="MLIR file, or '-' for stdin")
    args = parser.parse_args()

    try:
        print(cost_equation(read_input(args.input)))
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
