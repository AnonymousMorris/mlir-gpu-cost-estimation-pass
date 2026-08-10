from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import sympy

from .mlir_sympy import build_equations, parse_cost_function


@dataclass(frozen=True)
class BlockMetadata:
    num_ctas: int
    threads_per_block: int


def cost_equations(mlir: str) -> dict[str, sympy.Expr]:
    """Parse all named category expressions returned by a cost function."""
    return build_equations(parse_cost_function(mlir))


def block_metadata(mlir: str) -> BlockMetadata:
    """Parse and validate the cost function's block execution metadata."""
    cost_function = parse_cost_function(mlir)
    attributes = cost_function.operation.operation.attributes
    try:
        work_unit = attributes["cost.work_unit"].value
        metadata = BlockMetadata(
            num_ctas=int(attributes["cost.num_ctas"].value),
            threads_per_block=int(attributes["cost.threads_per_block"].value),
        )
    except KeyError as error:
        raise ValueError(f"cost function is missing {error.args[0]}") from error

    if work_unit != "block":
        raise ValueError(f"unsupported cost work unit: {work_unit}")
    if metadata.num_ctas <= 0:
        raise ValueError("cost.num_ctas must be positive")
    if metadata.threads_per_block <= 0:
        raise ValueError("cost.threads_per_block must be positive")
    return metadata


def _format_cost_equations(mlir: str) -> str:
    return "\n".join(
        f"{name}: {expression}"
        for name, expression in cost_equations(mlir).items()
    )


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simplify generated MLIR cost expressions"
    )
    parser.add_argument("input", help="MLIR file, or '-' for stdin")
    args = parser.parse_args()

    try:
        print(_format_cost_equations(_read_input(args.input)))
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
