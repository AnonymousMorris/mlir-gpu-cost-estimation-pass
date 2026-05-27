from __future__ import annotations

import pytest

from cost_analysis.dataset import Dataset, Sample, fit_global, metrics
from cost_analysis.parser import parse_cost_equation


def _sample(kernel: str, mlir: str, time_ms: float) -> Sample:
    return Sample(
        kernel=kernel,
        args=[],
        kwargs={},
        compiled_name=kernel,
        ttgir_filename=f"{kernel}.ttgir",
        time_ms=time_ms,
        equation=parse_cost_equation(mlir),
    )


def test_fit_global_recovers_linear_variable_values():
    x = """func.func @__cost_expr(%x: f64 {cost.name = "x"}, %y: f64 {cost.name = "y"}) -> f64 {
      %c2 = arith.constant 2.0 : f64
      %0 = arith.mulf %x, %c2 : f64
      %1 = arith.addf %0, %y : f64
      return %1 : f64
    }"""
    y = """func.func @__cost_expr(%x: f64 {cost.name = "x"}, %y: f64 {cost.name = "y"}) -> f64 {
      %c3 = arith.constant 3.0 : f64
      %0 = arith.mulf %y, %c3 : f64
      %1 = arith.addf %0, %x : f64
      return %1 : f64
    }"""
    dataset = Dataset(
        kernels={"a": [_sample("a", x, 5.0)], "b": [_sample("b", y, 5.0)]},
        variables=["x", "y"],
    )

    values = fit_global(dataset)

    assert values["x"] == pytest.approx(2.0)
    assert values["y"] == pytest.approx(1.0)
    assert metrics(dataset, values)["rmse"] == pytest.approx(0.0)


def test_fit_global_keeps_variable_values_nonnegative():
    mlir = """func.func @__cost_expr(%x: f64 {cost.name = "x"}) -> f64 {
      %c10 = arith.constant 10.0 : f64
      %0 = arith.addf %x, %c10 : f64
      return %0 : f64
    }"""
    dataset = Dataset(kernels={"a": [_sample("a", mlir, 5.0)]}, variables=["x"])

    values = fit_global(dataset)

    assert values["x"] == pytest.approx(0.0)
