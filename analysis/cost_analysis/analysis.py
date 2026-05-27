from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .config import PASS_PLUGIN
from .parser import CostEquation, parse_cost_equation


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalysisResult:
    equation: CostEquation
    raw_output: str


def analyze_ttgir(
    ttgir_path: Path,
    func_name: str,
    scalar_values: dict[str, float] | None = None,
) -> AnalysisResult:
    command = [
        "triton-opt",
        f"--load-pass-plugin={PASS_PLUGIN}",
        f"--pass-pipeline=builtin.module(my-cost-analysis{{func-name={func_name}}})",
        "-o",
        "/dev/null",
        str(ttgir_path),
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise AnalysisError(_first_lines(output) or f"triton-opt failed with {completed.returncode}")
    try:
        return AnalysisResult(
            equation=parse_cost_equation(output, scalar_values=scalar_values),
            raw_output=output,
        )
    except ValueError as exc:
        raise AnalysisError(str(exc)) from exc


def _first_lines(text: str, limit: int = 6) -> str:
    return "\n".join(text.splitlines()[:limit]).strip()
