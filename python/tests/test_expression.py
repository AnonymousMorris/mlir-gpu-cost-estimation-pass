from cost_ui.cost_function import CostFunction, CostOperation
from cost_ui.equation import build_cost_equation, evaluate_equation
from cost_ui.mlir_parser import CostParseError, parse_cost_function


SAMPLE_COST_MLIR = """func.func @__cost_expr(%dpas_cost: f64) -> f64 {
  %cst = arith.constant 2.200000e+01 : f64
  %cst_0 = arith.constant 3.100000e+01 : f64
  %cst_1 = arith.constant 1.280000e+02 : f64
  %cst_2 = arith.constant 0.000000e+00 : f64
  %cst_3 = arith.constant 2.700000e+01 : f64
  %0 = arith.addf %dpas_cost, %cst_3 : f64
  %1 = arith.mulf %dpas_cost, %cst_0 : f64
  %2 = arith.addf %0, %1 : f64
  %3 = arith.addf %2, %cst_2 : f64
  %4 = arith.mulf %3, %cst_1 : f64
  %5 = arith.addf %4, %cst_3 : f64
  %6 = arith.addf %5, %cst : f64
  %7 = arith.addf %6, %cst_2 : f64
  return %7 : f64
}"""


def test_parse_and_evaluate_sample_cost_expression(monkeypatch):
    monkeypatch.setattr("cost_ui.mlir_parser._parse_with_mlir_binding", lambda text: text)

    cost_function = parse_cost_function(SAMPLE_COST_MLIR)
    equation = build_cost_equation(cost_function)

    assert equation.variables == ["dpas_cost"]
    assert equation.javascript_parameters == ["v0"]
    assert equation.equation_text == "4096.0*dpas_cost + 3505.0"
    assert equation.javascript_expression == "4096.0*v0 + 3505.0"
    assert evaluate_equation(equation, {"dpas_cost": 1}) == 7601
    assert evaluate_equation(equation, {"dpas_cost": 2}) == 11697


def test_build_equation_from_cost_function_ir():
    cost_function = CostFunction(
        variables=["x"],
        operations=[
            CostOperation(result="%c0", kind="constant", value=2),
            CostOperation(result="%0", kind="mulf", operands=("%x", "%c0")),
            CostOperation(result="%1", kind="addf", operands=("%0", "%x")),
        ],
        return_value="%1",
    )

    equation = build_cost_equation(cost_function)

    assert equation.equation_text == "3.0*x"
    assert equation.javascript_expression == "3.0*v0"
    assert evaluate_equation(equation, {"x": 4}) == 12


def test_dotted_variable_names_get_safe_javascript_parameters(monkeypatch):
    monkeypatch.setattr("cost_ui.mlir_parser._parse_with_mlir_binding", lambda text: None)
    mlir = """func.func @__cost_expr(%residency.p1: f64, %residency.l1_cost: f64) -> f64 {
      %0 = arith.mulf %residency.p1, %residency.l1_cost : f64
      return %0 : f64
    }"""

    equation = build_cost_equation(parse_cost_function(mlir))

    assert equation.variables == ["residency.p1", "residency.l1_cost"]
    assert equation.javascript_parameters == ["v0", "v1"]
    assert equation.equation_text == "residency.l1_cost*residency.p1"
    assert equation.javascript_expression == "v0*v1"


def test_ignores_non_mlir_banner_before_cost_function(monkeypatch):
    monkeypatch.setattr("cost_ui.mlir_parser._parse_with_mlir_binding", lambda text: None)
    mlir = """/ Cost expression for func.func @main
func.func @__cost_expr(%x: f64) -> f64 {
  return %x : f64
}"""

    equation = build_cost_equation(parse_cost_function(mlir))

    assert equation.variables == ["x"]
    assert equation.javascript_expression == "v0"


def test_rejects_missing_cost_function(monkeypatch):
    monkeypatch.setattr("cost_ui.mlir_parser._parse_with_mlir_binding", lambda text: text)

    try:
        parse_cost_function("module {}")
    except CostParseError as exc:
        assert "@__cost_expr" in str(exc)
    else:
        raise AssertionError("expected parse failure")


def test_rejects_unsupported_operation(monkeypatch):
    monkeypatch.setattr("cost_ui.mlir_parser._parse_with_mlir_binding", lambda text: text)

    mlir = """func.func @__cost_expr(%x: f64) -> f64 {
      %0 = arith.divf %x, %x : f64
      return %0 : f64
    }"""

    try:
        parse_cost_function(mlir)
    except CostParseError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected parse failure")
