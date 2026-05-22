from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy.printing.jscode import jscode

from .cost_function import CostFunction
from .mlir_parser import CostParseError


@dataclass(frozen=True)
class CostEquation:
    variables: list[str]
    javascript_parameters: list[str]
    expression: sympy.Expr
    equation_text: str
    javascript_expression: str


def build_cost_equation(cost_function: CostFunction) -> CostEquation:
    display_symbols = {
        name: sympy.Symbol(name) for name in cost_function.variables
    }
    javascript_symbols = {
        display_symbols[name]: sympy.Symbol(_javascript_parameter_name(index))
        for index, name in enumerate(cost_function.variables)
    }
    values: dict[str, sympy.Expr] = {
        f"%{name}": display_symbols[name] for name in cost_function.variables
    }

    for operation in cost_function.operations:
        if operation.kind == "constant":
            if operation.value is None:
                raise CostParseError(f"constant {operation.result} has no value")
            values[operation.result] = sympy.Float(operation.value)
            continue

        lhs = _lookup(operation.operands[0], values)
        rhs = _lookup(operation.operands[1], values)
        if operation.kind == "addf":
            values[operation.result] = lhs + rhs
        elif operation.kind == "subf":
            values[operation.result] = lhs - rhs
        elif operation.kind == "mulf":
            values[operation.result] = lhs * rhs
        else:
            raise CostParseError(f"unsupported cost operation kind: {operation.kind}")

    expression = _normalize_expression(_lookup(cost_function.return_value, values))
    javascript_expression = expression.xreplace(javascript_symbols)
    return CostEquation(
        variables=cost_function.variables,
        javascript_parameters=[
            _javascript_parameter_name(index)
            for index, _ in enumerate(cost_function.variables)
        ],
        expression=expression,
        equation_text=str(expression),
        javascript_expression=jscode(javascript_expression),
    )


def evaluate_equation(equation: CostEquation, values: dict[str, float]) -> float:
    missing = [name for name in equation.variables if name not in values]
    if missing:
        raise ValueError(f"missing value for variable '{missing[0]}'")

    substitutions = {
        sympy.Symbol(name): float(values[name]) for name in equation.variables
    }
    return float(equation.expression.evalf(subs=substitutions))


def _lookup(name: str, values: dict[str, sympy.Expr]) -> sympy.Expr:
    try:
        return values[name]
    except KeyError as exc:
        raise CostParseError(f"operation references unknown value {name}") from exc


def _normalize_expression(expression: sympy.Expr) -> sympy.Expr:
    return sympy.simplify(expression)


def _javascript_parameter_name(index: int) -> str:
    return f"v{index}"
