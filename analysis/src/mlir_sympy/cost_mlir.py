from __future__ import annotations

from dataclasses import dataclass

from mlir import ir
from mlir.dialects import func


COST_ARGUMENT_NAME_ATTR = "cost.name"
COST_ARGUMENT_KIND_ATTR = "cost.kind"


@dataclass(frozen=True)
class CostFunction:
    context: ir.Context
    module: ir.Module
    operation: func.FuncOp
    argument_names: list[str]
    argument_kinds: list[str | None]
    result_names: list[str]


def parse_cost_function(text: str) -> CostFunction:
    if not text.strip():
        raise ValueError("input MLIR is empty")

    context = ir.Context()
    context.allow_unregistered_dialects = True
    module = ir.Module.parse(text, context=context)
    operation = find_cost_function(module)
    return CostFunction(
        context=context,
        module=module,
        operation=operation,
        argument_names=argument_names(operation),
        argument_kinds=argument_kinds(operation),
        result_names=result_names(operation),
    )


def find_cost_function(module: ir.Module) -> func.FuncOp:
    return next(iter(module.body.operations))


def argument_names(operation: func.FuncOp) -> list[str]:
    names: list[str] = []
    for index, _ in enumerate(operation.arguments):
        names.append(operation.arg_attrs[index][COST_ARGUMENT_NAME_ATTR].value)
    return names


def argument_kinds(operation: func.FuncOp) -> list[str | None]:
    kinds: list[str | None] = []
    for index, _ in enumerate(operation.arguments):
        try:
            kinds.append(
                operation.arg_attrs[index][COST_ARGUMENT_KIND_ATTR].value
            )
        except KeyError:
            kinds.append(None)
    return kinds


def result_names(operation: func.FuncOp) -> list[str]:
    try:
        result_attrs = operation.result_attrs
    except KeyError:
        return [f"result{index}" for index, _ in enumerate(operation.type.results)]

    names: list[str] = []
    for index, attrs in enumerate(result_attrs):
        try:
            names.append(attrs[COST_ARGUMENT_NAME_ATTR].value)
        except KeyError:
            names.append(f"result{index}")
    return names
