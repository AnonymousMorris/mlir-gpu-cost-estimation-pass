from __future__ import annotations

from dataclasses import dataclass
import re

import sympy


class CostParseError(ValueError):
    pass


@dataclass(frozen=True)
class CostEquation:
    variables: list[str]
    expression: sympy.Expr
    equation_text: str
    coefficients: dict[str, float]
    constant: float
    mlir: str


_FUNC_RE = re.compile(
    r"func\.func\s+@__cost_expr\s*\((?P<args>.*?)\)\s*->\s*f64\s*\{(?P<body>.*?)^\s*\}",
    re.DOTALL | re.MULTILINE,
)
_ARG_RE = re.compile(
    r"%(?P<ssa>[A-Za-z_.$-][\w.$-]*)\s*:\s*(?P<type>[^,{)]+)(?:\s*\{(?P<attrs>[^}]*)\})?"
)
_COST_NAME_RE = re.compile(r'cost\.name\s*=\s*"(?P<name>[^"]+)"')
_ASSIGN_RE = re.compile(r"^\s*(?P<result>%[\w.$-]+)\s*=\s*(?P<op>.+?)\s*$")
_CONST_RE = re.compile(
    r"arith\.constant\s+(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*:\s*f64$"
)
_BINARY_RE = re.compile(
    r"arith\.(?P<kind>addf|subf|mulf)\s+(?P<lhs>%[\w.$-]+)\s*,\s*(?P<rhs>%[\w.$-]+)\s*:\s*f64$"
)
_RETURN_RE = re.compile(r"^\s*return\s+(?P<value>%[\w.$-]+)\s*:\s*f64\s*$")
_VALID_VARIABLE_RE = re.compile(r"^[A-Za-z_][\w.:-]*$")


def parse_cost_equation(
    mlir_text: str,
    scalar_values: dict[str, float] | None = None,
) -> CostEquation:
    scalar_values = scalar_values or {}
    match = _single_cost_function(mlir_text)
    variables, values = _parse_arguments(match.group("args"), scalar_values)

    returned_name: str | None = None
    for raw_line in match.group("body").splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        return_match = _RETURN_RE.match(line)
        if return_match:
            returned_name = return_match.group("value")
            continue

        assign_match = _ASSIGN_RE.match(line)
        if not assign_match:
            raise CostParseError(f"unsupported cost expression line: {line}")

        result = assign_match.group("result")
        values[result] = _parse_operation(assign_match.group("op"), values)

    if returned_name is None:
        raise CostParseError("@__cost_expr does not return a value")
    if returned_name not in values:
        raise CostParseError(f"return references unknown value {returned_name}")

    expression = sympy.simplify(values[returned_name])
    coefficients, constant = _linear_terms(expression, variables)
    return CostEquation(
        variables=variables,
        expression=expression,
        equation_text=str(expression),
        coefficients=coefficients,
        constant=constant,
        mlir=match.group(0),
    )


def _single_cost_function(mlir_text: str) -> re.Match[str]:
    matches = list(_FUNC_RE.finditer(mlir_text))
    if not matches:
        raise CostParseError("could not find func.func @__cost_expr returning f64")
    if len(matches) > 1:
        raise CostParseError("found multiple func.func @__cost_expr definitions")
    return matches[0]


def _parse_arguments(
    args_text: str,
    scalar_values: dict[str, float],
) -> tuple[list[str], dict[str, sympy.Expr]]:
    variables: list[str] = []
    values: dict[str, sympy.Expr] = {}
    parsed_argument = False
    for match in _ARG_RE.finditer(args_text):
        parsed_argument = True
        ssa_name = f"%{match.group('ssa')}"
        arg_type = match.group("type").strip()
        attrs = match.group("attrs") or ""
        cost_name_match = _COST_NAME_RE.search(attrs)
        display_name = cost_name_match.group("name") if cost_name_match else match.group("ssa")

        if arg_type != "f64":
            if display_name not in scalar_values:
                raise CostParseError(
                    f"missing scalar value for non-f64 cost argument '{display_name}' "
                    f"of type {arg_type}"
                )
            values[ssa_name] = _numeric_literal(scalar_values[display_name])
            continue
        if not _VALID_VARIABLE_RE.match(display_name):
            raise CostParseError(f"unsupported cost variable name '{display_name}'")

        variables.append(display_name)
        values[ssa_name] = sympy.Symbol(display_name)

    if not parsed_argument and args_text.strip():
        raise CostParseError("only f64 cost arguments are supported in @__cost_expr")
    return variables, values


def _parse_operation(op_text: str, values: dict[str, sympy.Expr]) -> sympy.Expr:
    const_match = _CONST_RE.match(op_text)
    if const_match:
        return sympy.Float(float(const_match.group("value")))

    int_const_match = re.match(r"arith\.constant\s+(?P<value>[-+]?\d+)\s*:\s*(?:i\d+|index)$", op_text)
    if int_const_match:
        return sympy.Integer(int(int_const_match.group("value")))

    binary_match = _BINARY_RE.match(op_text)
    if binary_match:
        lhs = _lookup(binary_match.group("lhs"), values)
        rhs = _lookup(binary_match.group("rhs"), values)
        kind = binary_match.group("kind")
        if kind == "addf":
            return lhs + rhs
        if kind == "subf":
            return lhs - rhs
        if kind == "mulf":
            return lhs * rhs

    int_binary_match = re.match(
        r"arith\.(?P<kind>addi|subi|muli|divsi|divui|remsi|remui|ceildivui)\s+"
        r"(?P<lhs>%[\w.$-]+)\s*,\s*(?P<rhs>%[\w.$-]+)\s*:\s*(?:i\d+|index)$",
        op_text,
    )
    if int_binary_match:
        lhs = _lookup(int_binary_match.group("lhs"), values)
        rhs = _lookup(int_binary_match.group("rhs"), values)
        kind = int_binary_match.group("kind")
        if kind == "addi":
            return lhs + rhs
        if kind == "subi":
            return lhs - rhs
        if kind == "muli":
            return lhs * rhs
        if kind in {"divsi", "divui"}:
            return sympy.floor(lhs / rhs)
        if kind in {"remsi", "remui"}:
            return sympy.Mod(lhs, rhs)
        if kind == "ceildivui":
            return sympy.ceiling(lhs / rhs)

    cast_match = re.match(
        r"arith\.(?:uitofp|sitofp)\s+(?P<value>%[\w.$-]+)\s*:\s*(?:i\d+|index)\s+to\s+f64$",
        op_text,
    )
    if cast_match:
        return sympy.Float(_lookup(cast_match.group("value"), values))

    raise CostParseError(f"unsupported cost expression op: {op_text}")


def _lookup(name: str, values: dict[str, sympy.Expr]) -> sympy.Expr:
    try:
        return values[name]
    except KeyError as exc:
        raise CostParseError(f"operation references unknown value {name}") from exc


def _linear_terms(expression: sympy.Expr, variables: list[str]) -> tuple[dict[str, float], float]:
    if not variables:
        return {}, float(expression)

    symbols = [sympy.Symbol(name) for name in variables]
    try:
        poly = sympy.Poly(sympy.expand(expression), *symbols)
    except sympy.PolynomialError as exc:
        raise CostParseError("cost expression is not polynomial in cost variables") from exc
    if poly.total_degree() > 1:
        raise CostParseError("cost expression is not linear in cost variables")

    coefficients: dict[str, float] = {}
    for name, symbol in zip(variables, symbols, strict=True):
        coefficients[name] = float(poly.coeff_monomial(symbol))
    return coefficients, float(poly.coeff_monomial(1))


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0]


def _numeric_literal(value: float) -> sympy.Expr:
    numeric = float(value)
    if numeric.is_integer():
        return sympy.Integer(int(numeric))
    return sympy.Float(numeric)
