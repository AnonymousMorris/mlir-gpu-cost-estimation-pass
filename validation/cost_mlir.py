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
    )


def find_cost_function(module: ir.Module) -> func.FuncOp:
    return next(iter(module.body.operations))


def argument_names(operation: func.FuncOp) -> list[str]:
    names: list[str] = []
    for index, _ in enumerate(operation.arguments):
        names.append(operation.arg_attrs[index][COST_ARGUMENT_NAME_ATTR].value)
    return names
