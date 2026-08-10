from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .scheduler import Schedule


@dataclass(frozen=True)
class CostResult:
    kernel: str
    ttgir: str
    time_ms: float
    predicted_ms: float
    bottleneck: str
    scheduled_work: dict[str, float]
    pipeline_ms: dict[str, float]
    expression: str
    category_expressions: dict[str, str]
    args: list[str]
    kwargs: dict[str, str]
    grid_size: list[int]
    schedule: Schedule


@dataclass(frozen=True)
class AnalysisSkip:
    kernel: str
    ttgir: str
    reason: str


def write_output(
    path: Path,
    rows: list[CostResult],
    skips: list[AnalysisSkip],
) -> None:
    payload = {
        "predictions": [asdict(row) for row in rows],
        "analysis_skips": [asdict(skip) for skip in skips],
        "summary": summarize(rows, skips),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize(
    rows: list[CostResult],
    skips: list[AnalysisSkip],
) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "skip_count": len(skips)}
    ratios = [row.predicted_ms / row.time_ms for row in rows if row.time_ms > 0]
    factors = [max(ratio, 1.0 / ratio) for ratio in ratios if ratio > 0]
    return {
        "count": len(rows),
        "skip_count": len(skips),
        "median_predicted_over_actual": median(ratios),
        "mean_predicted_over_actual": sum(ratios) / len(ratios),
        "median_multiplicative_error": median(factors),
        "within_2x_fraction": sum(factor <= 2.0 for factor in factors) / len(factors),
        "within_5x_fraction": sum(factor <= 5.0 for factor in factors) / len(factors),
    }

