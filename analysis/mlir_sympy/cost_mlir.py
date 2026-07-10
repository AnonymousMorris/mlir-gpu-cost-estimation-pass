from __future__ import annotations

from dataclasses import dataclass

from mlir import ir
from mlir.dialects import func


COST_ARGUMENT_NAME_ATTR = "cost.name"


@dataclass(frozen=True)
class CostFunction:
    context: ir.Context
    module: ir.Module
    operation: func.FuncOp
    argument_names: list[str]
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
        result_names=result_names(operation),
    )


def find_cost_function(module: ir.Module) -> func.FuncOp:
    return next(iter(module.body.operations))


def argument_names(operation: func.FuncOp) -> list[str]:
    names: list[str] = []
    for index, _ in enumerate(operation.arguments):
        names.append(operation.arg_attrs[index][COST_ARGUMENT_NAME_ATTR].value)
    return names


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
