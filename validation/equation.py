from __future__ import annotations

from mlir import ir
from mlir.dialects import arith, func
import sympy

from cost_mlir import CostFunction


def build_equation(cost_function: CostFunction) -> sympy.Expr:
    block = entry_block(cost_function.operation)
    values = bind_arguments(block, cost_function.argument_names)

    for op in block.operations:
        if isinstance(op, func.ReturnOp):
            return sympy.simplify(values[op.operands[0]])

        if isinstance(op, arith.ConstantOp):
            values[op.result] = sympy.Float(op.literal_value)
            continue

        if isinstance(op, arith.AddFOp):
            values[op.result] = values[op.lhs] + values[op.rhs]
            continue

        if isinstance(op, arith.SubFOp):
            values[op.result] = values[op.lhs] - values[op.rhs]
            continue

        if isinstance(op, arith.MulFOp):
            values[op.result] = values[op.lhs] * values[op.rhs]
            continue

        raise ValueError(f"unsupported op: {op.operation.name}")

    raise ValueError("@__cost_expr does not return a value")


def entry_block(operation: func.FuncOp) -> ir.Block:
    return next(iter(operation.regions[0].blocks))


def bind_arguments(block: ir.Block, names: list[str]) -> dict[ir.Value, sympy.Expr]:
    values: dict[ir.Value, sympy.Expr] = {}
    for index, argument in enumerate(block.arguments):
        name = names[index] if index < len(names) else f"arg{index}"
        values[argument] = sympy.Symbol(name)
    return values
