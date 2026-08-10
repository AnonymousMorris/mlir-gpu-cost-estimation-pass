from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import sympy

from .mlir_sympy import CostFunction, build_equations, parse_cost_function


@dataclass(frozen=True)
class BlockMetadata:
    num_ctas: int
    threads_per_block: int


@dataclass(frozen=True)
class ParsedCost:
    equations: dict[str, sympy.Expr]
    metadata: BlockMetadata
    weight_symbols: frozenset[str]
    runtime_symbols: frozenset[str]


def cost_equations(mlir: str) -> dict[str, sympy.Expr]:
    """Parse all named category expressions returned by a cost function."""
    return build_equations(parse_cost_function(mlir))


def _block_metadata(cost_function: CostFunction) -> BlockMetadata:
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


def block_metadata(mlir: str) -> BlockMetadata:
    """Parse and validate the cost function's block execution metadata."""
    return _block_metadata(parse_cost_function(mlir))


def parse_cost_analysis(mlir: str) -> ParsedCost:
    """Parse equations, execution metadata, and symbolic input roles."""
    cost_function = parse_cost_function(mlir)
    kinds = dict(zip(cost_function.argument_names, cost_function.argument_kinds))
    if len(kinds) != len(cost_function.argument_names):
        raise ValueError("cost argument names must be unique")

    missing_kinds = sorted(name for name, kind in kinds.items() if kind is None)
    if missing_kinds:
        raise ValueError(
            "cost arguments are missing cost.kind: " + ", ".join(missing_kinds)
        )
    invalid_kinds = sorted(
        f"{name}={kind}" for name, kind in kinds.items()
        if kind not in {"runtime", "weight"}
    )
    if invalid_kinds:
        raise ValueError(
            "cost arguments have invalid cost.kind: " + ", ".join(invalid_kinds)
        )

    return ParsedCost(
        equations=build_equations(cost_function),
        metadata=_block_metadata(cost_function),
        weight_symbols=frozenset(
            name for name, kind in kinds.items() if kind == "weight"
        ),
        runtime_symbols=frozenset(
            name for name, kind in kinds.items() if kind == "runtime"
        ),
    )


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
