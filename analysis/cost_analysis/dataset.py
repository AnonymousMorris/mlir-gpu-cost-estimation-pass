from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .analysis import analyze_ttgir
from .config import RESULT_JSON, TTGIR_ROOT
from .parser import CostEquation


DEFAULT_VARIABLE_VALUE = 1.0


@dataclass(frozen=True)
class Sample:
    kernel: str
    args: list[str]
    kwargs: dict[str, str]
    compiled_name: str
    ttgir_filename: str
    time_ms: float
    equation: CostEquation

    @property
    def label(self) -> str:
        args = ", ".join(self.args)
        kwargs = ", ".join(f"{key}={value}" for key, value in self.kwargs.items())
        if args and kwargs:
            return f"{self.compiled_name}({args}; {kwargs})"
        if args:
            return f"{self.compiled_name}({args})"
        return f"{self.compiled_name}({kwargs})"


@dataclass
class Dataset:
    kernels: dict[str, list[Sample]]
    variables: list[str]


class DatasetStore:
    def __init__(
        self,
        result_json: Path = RESULT_JSON,
        ttgir_root: Path = TTGIR_ROOT,
    ) -> None:
        self._result_json = result_json
        self._ttgir_root = ttgir_root
        self._lock = Lock()
        self._dataset: Dataset | None = None

    def load(self, refresh: bool = False) -> Dataset:
        with self._lock:
            if self._dataset is None or refresh:
                self._dataset = self._build()
            return self._dataset

    def _build(self) -> Dataset:
        raw: dict[str, list[dict[str, Any]]] = json.loads(self._result_json.read_text())
        kernels: dict[str, list[Sample]] = {}
        variables: set[str] = set()
        for kernel, rows in raw.items():
            samples: list[Sample] = []
            for row in rows:
                analysis = analyze_ttgir(
                    self._ttgir_root / row["ttgir_filename"],
                    row["compiled_name"],
                    scalar_values=_scalar_values(
                        self._ttgir_root / row["ttgir_filename"],
                        row["compiled_name"],
                        list(row.get("args", [])),
                    ),
                )
                equation = analysis.equation
                variables.update(equation.variables)
                samples.append(
                    Sample(
                        kernel=kernel,
                        args=list(row.get("args", [])),
                        kwargs=dict(row.get("kwargs", {})),
                        compiled_name=row["compiled_name"],
                        ttgir_filename=row["ttgir_filename"],
                        time_ms=float(row["time_ms"]),
                        equation=equation,
                    )
                )
            kernels[kernel] = samples
        return Dataset(kernels=kernels, variables=sorted(variables))


def default_values(dataset: Dataset) -> dict[str, float]:
    return {name: DEFAULT_VARIABLE_VALUE for name in dataset.variables}


def estimate(sample: Sample, values: dict[str, float]) -> float:
    total = sample.equation.constant
    for variable, coefficient in sample.equation.coefficients.items():
        total += coefficient * float(values.get(variable, DEFAULT_VARIABLE_VALUE))
    return total


def fit_global(dataset: Dataset) -> dict[str, float]:
    samples = [sample for rows in dataset.kernels.values() for sample in rows]
    if not samples:
        return default_values(dataset)

    variables = dataset.variables
    matrix = np.array(
        [
            [sample.equation.coefficients.get(variable, 0.0) for variable in variables]
            for sample in samples
        ],
        dtype=float,
    )
    target = np.array(
        [sample.time_ms - sample.equation.constant for sample in samples],
        dtype=float,
    )
    if matrix.size == 0:
        return {}
    solution = _nonnegative_lstsq(matrix, target)
    return {variable: float(value) for variable, value in zip(variables, solution, strict=True)}


def _nonnegative_lstsq(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    _, variable_count = matrix.shape
    if variable_count == 0:
        return np.array([], dtype=float)

    passive = np.zeros(variable_count, dtype=bool)
    solution = np.zeros(variable_count, dtype=float)
    gradient = matrix.T @ (target - matrix @ solution)
    tolerance = 10 * np.finfo(float).eps * max(np.linalg.norm(matrix, 1), 1.0) * (variable_count + 1)

    for _ in range(max(1, 30 * variable_count)):
        if not np.any((~passive) & (gradient > tolerance)):
            break

        active_index = int(np.argmax(np.where(~passive, gradient, -np.inf)))
        passive[active_index] = True

        while True:
            candidate = np.zeros(variable_count, dtype=float)
            candidate[passive], *_ = np.linalg.lstsq(matrix[:, passive], target, rcond=None)
            if np.all(candidate[passive] > tolerance):
                solution = candidate
                break

            blocking = passive & (candidate <= tolerance)
            alpha = np.min(solution[blocking] / (solution[blocking] - candidate[blocking]))
            solution = solution + alpha * (candidate - solution)
            passive[passive & (solution <= tolerance)] = False

        gradient = matrix.T @ (target - matrix @ solution)

    solution[solution < tolerance] = 0.0
    return solution


def metrics(dataset: Dataset, values: dict[str, float]) -> dict[str, float]:
    samples = [sample for rows in dataset.kernels.values() for sample in rows]
    if not samples:
        return {"rmse": 0.0, "mae": 0.0}
    errors = np.array([estimate(sample, values) - sample.time_ms for sample in samples], dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
    }


_FUNC_RE = re.compile(
    r"tt\.func\s+(?:public\s+)?@(?P<name>[\w.$-]+)\s*\((?P<args>.*?)\)\s*attributes",
    re.DOTALL,
)
_ARG_RE = re.compile(r"%(?P<name>[A-Za-z_.$-][\w.$-]*)\s*:\s*(?P<type>[^,{)]+)")


def _scalar_values(ttgir_path: Path, func_name: str, benchmark_args: list[str]) -> dict[str, float]:
    text = ttgir_path.read_text()
    scalar_names: list[str] = []
    for match in _FUNC_RE.finditer(text):
        if match.group("name") != func_name:
            continue
        for arg_match in _ARG_RE.finditer(match.group("args")):
            arg_type = arg_match.group("type").strip()
            if not arg_type.startswith("!tt.ptr"):
                scalar_names.append(arg_match.group("name"))
        break

    values: dict[str, float] = {}
    for name, raw_value in zip(scalar_names, benchmark_args, strict=False):
        try:
            values[name] = float(raw_value)
        except ValueError:
            continue
    return values
