from __future__ import annotations

from mlir import ir
from mlir.dialects import arith, func
import sympy

from mlir_sympy.cost_mlir import CostFunction


def build_equations(cost_function: CostFunction) -> dict[str, sympy.Expr]:
    block = entry_block(cost_function.operation)
    values = bind_arguments(block, cost_function.argument_names)

    for op in block.operations:
        if isinstance(op, func.ReturnOp):
            if len(op.operands) != len(cost_function.result_names):
                raise ValueError("cost result names do not match returned values")
            equations = {
                name: sympy.simplify(values[value])
                for name, value in zip(cost_function.result_names, op.operands)
            }
            if len(equations) != len(op.operands):
                raise ValueError("cost result names must be unique")
            return equations

        if isinstance(op, arith.ConstantOp):
            values[op.result] = sympy.Float(op.literal_value)
            continue

        if isinstance(op, arith.AddFOp):
            values[op.result] = values[op.lhs] + values[op.rhs]
            continue

        if isinstance(op, arith.AddIOp):
            values[op.result] = values[op.lhs] + values[op.rhs]
            continue

        if isinstance(op, arith.SubFOp):
            values[op.result] = values[op.lhs] - values[op.rhs]
            continue

        if isinstance(op, arith.SubIOp):
            values[op.result] = values[op.lhs] - values[op.rhs]
            continue

        if isinstance(op, arith.MulFOp):
            values[op.result] = values[op.lhs] * values[op.rhs]
            continue

        if isinstance(op, arith.MulIOp):
            values[op.result] = values[op.lhs] * values[op.rhs]
            continue

        if isinstance(op, (arith.DivSIOp, arith.DivUIOp)):
            values[op.result] = sympy.floor(values[op.lhs] / values[op.rhs])
            continue

        if isinstance(op, (arith.CeilDivSIOp, arith.CeilDivUIOp)):
            values[op.result] = sympy.ceiling(values[op.lhs] / values[op.rhs])
            continue

        if isinstance(op, (arith.IndexCastOp, arith.IndexCastUIOp, arith.ExtUIOp, arith.TruncIOp, arith.UIToFPOp)):
            values[op.result] = values[op.operands[0]]
            continue

        if isinstance(op, arith.MaximumFOp):
            values[op.result] = sympy.Max(values[op.lhs], values[op.rhs])
            continue

        raise ValueError(f"unsupported op: {op.operation.name}")

    raise ValueError("@__cost_expr does not return values")


def build_equation(cost_function: CostFunction) -> sympy.Expr:
    equations = build_equations(cost_function)
    if len(equations) != 1:
        raise ValueError("@__cost_expr returns multiple cost categories")
    return next(iter(equations.values()))


def entry_block(operation: func.FuncOp) -> ir.Block:
    return next(iter(operation.regions[0].blocks))


def bind_arguments(block: ir.Block, names: list[str]) -> dict[ir.Value, sympy.Expr]:
    values: dict[ir.Value, sympy.Expr] = {}
    for index, argument in enumerate(block.arguments):
        name = names[index] if index < len(names) else f"arg{index}"
        values[argument] = sympy.Symbol(name)
    return values
