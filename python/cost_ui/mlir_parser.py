from __future__ import annotations

import re

from .cost_function import CostFunction, CostOperation
from .mlir_binding import MlirBindingError, load_mlir_ir


class CostParseError(ValueError):
    pass


_FUNC_RE = re.compile(
    r"func\.func\s+@__cost_expr\s*\((?P<args>.*?)\)\s*->\s*f64\s*\{(?P<body>.*?)^\s*\}",
    re.DOTALL | re.MULTILINE,
)
_ARG_RE = re.compile(r"%(?P<name>[A-Za-z_.$-][\w.$-]*)\s*:\s*f64")
_ASSIGN_RE = re.compile(r"^\s*(?P<result>%[\w.$-]+)\s*=\s*(?P<op>.+?)\s*$")
_CONST_RE = re.compile(
    r"arith\.constant\s+(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*:\s*f64$"
)
_BINARY_RE = re.compile(
    r"(?P<op>arith\.(?:addf|subf|mulf))\s+(?P<lhs>%[\w.$-]+)\s*,\s*(?P<rhs>%[\w.$-]+)\s*:\s*f64$"
)
_RETURN_RE = re.compile(r"^\s*return\s+(?P<value>%[\w.$-]+)\s*:\s*f64\s*$")


def parse_cost_function(mlir_text: str) -> CostFunction:
    """Parse generated cost MLIR into a small arithmetic IR.

    The MLIR Python binding is the parser of record: input is first parsed with
    `mlir.ir.Module.parse`, and the cost function is then extracted from the
    parsed module assembly. The remaining conversion is intentionally narrow and
    only accepts the arithmetic emitted by the cost-analysis pass.
    """

    cost_func_text = _extract_cost_function_text(mlir_text)
    _parse_with_mlir_binding(cost_func_text)

    matches = list(_FUNC_RE.finditer(cost_func_text))
    if not matches:
        raise CostParseError("could not find func.func @__cost_expr returning f64")
    if len(matches) > 1:
        raise CostParseError("found multiple func.func @__cost_expr definitions")

    match = matches[0]
    variables = _parse_arguments(match.group("args"))
    known_values = {f"%{name}" for name in variables}

    returned_name: str | None = None
    operations: list[CostOperation] = []
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

        result_name = assign_match.group("result")
        operation = _parse_operation(result_name, assign_match.group("op"), known_values)
        operations.append(operation)
        known_values.add(result_name)

    if returned_name is None:
        raise CostParseError("@__cost_expr does not return a value")
    if returned_name not in known_values:
        raise CostParseError(f"return references unknown value {returned_name}")

    return CostFunction(
        variables=variables,
        operations=operations,
        return_value=returned_name,
    )


def _parse_with_mlir_binding(mlir_text: str) -> None:
    try:
        Context, Module = load_mlir_ir()
    except MlirBindingError as exc:
        raise CostParseError(str(exc)) from exc

    parse_text = mlir_text if "module" in mlir_text else f"module {{\n{mlir_text}\n}}"
    try:
        with Context() as ctx:
            ctx.allow_unregistered_dialects = True
            Module.parse(parse_text)
    except Exception as exc:
        raise CostParseError(f"MLIR parser rejected input: {exc}") from exc


def _extract_cost_function_text(mlir_text: str) -> str:
    matches = list(_FUNC_RE.finditer(mlir_text))
    if not matches:
        raise CostParseError("could not find func.func @__cost_expr returning f64")
    if len(matches) > 1:
        raise CostParseError("found multiple func.func @__cost_expr definitions")
    return matches[0].group(0)


def _parse_arguments(args_text: str) -> list[str]:
    variables = [match.group("name") for match in _ARG_RE.finditer(args_text)]
    if not variables and args_text.strip():
        raise CostParseError("only f64 arguments are supported in @__cost_expr")
    return variables


def _parse_operation(
    result_name: str, op_text: str, known_values: set[str]
) -> CostOperation:
    const_match = _CONST_RE.match(op_text)
    if const_match:
        return CostOperation(
            result=result_name,
            kind="constant",
            value=float(const_match.group("value")),
        )

    binary_match = _BINARY_RE.match(op_text)
    if binary_match:
        kind = {"arith.addf": "addf", "arith.subf": "subf", "arith.mulf": "mulf"}[
            binary_match.group("op")
        ]
        lhs = binary_match.group("lhs")
        rhs = binary_match.group("rhs")
        _require_known_value(lhs, known_values)
        _require_known_value(rhs, known_values)
        return CostOperation(
            result=result_name,
            kind=kind,
            operands=(lhs, rhs),
        )

    raise CostParseError(f"unsupported cost expression op: {op_text}")


def _require_known_value(name: str, known_values: set[str]) -> None:
    if name not in known_values:
        raise CostParseError(f"operation references unknown value {name}")


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0]
