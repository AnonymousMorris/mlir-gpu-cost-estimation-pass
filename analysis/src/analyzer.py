from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import sympy

from .cost_model import (
    eval_work,
    grid_program_count,
    per_sm_throughput_rates,
    pipeline_exprs,
    predict_ms,
)
from .cost_pass import run_cost_pass
from .parse import block_metadata, cost_equations
from .results import AnalysisSkip, CostResult
from .scheduler import schedule_work


ROOT = Path(__file__).resolve().parent.parent
GPU_SPEC_PATH = ROOT / "gpu_spec.json"
MODEL_CONFIG_PATH = ROOT / "cost_analysis_config.json"
TRITON_OPT = Path("triton-opt")
PASS_PLUGIN = ROOT.parent / "build" / "libMyPass.so"
PASS_TIMEOUT_S = 10.0


def load_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def iter_records(
    results: dict[str, list[dict[str, Any]]],
) -> Iterator[tuple[str, dict[str, Any]]]:
    for kernel, records in results.items():
        for record in records:
            if record.get("status") == "ok" and record.get("ttgir_filename"):
                yield kernel, record


def analyze(
    results_path: Path,
    ttgir_dir: Path,
) -> tuple[list[CostResult], list[AnalysisSkip]]:
    results = load_json(results_path)
    spec = load_json(GPU_SPEC_PATH)
    config = load_json(MODEL_CONFIG_PATH)
    rates = per_sm_throughput_rates(spec, config)
    num_sms = int(spec["sms"])
    launch_overhead_ms = float(spec.get("launch_overhead_ms", 0.0))

    rows: list[CostResult] = []
    skips: list[AnalysisSkip] = []
    for kernel, record in iter_records(results):
        ttgir = record["ttgir_filename"]
        ttgir_path = ttgir_dir / ttgir
        func_name = record.get("compiled_name") or kernel
        if not ttgir_path.exists():
            skips.append(AnalysisSkip(kernel, ttgir, "missing ttgir"))
            continue

        try:
            grid_size = [int(value) for value in record.get("grid_size", [])]
            programs = grid_program_count(grid_size)
            cost_mlir = run_cost_pass(
                TRITON_OPT,
                PASS_PLUGIN,
                ttgir_path,
                func_name,
                PASS_TIMEOUT_S,
            )
            exprs = pipeline_exprs(cost_equations(cost_mlir))
            metadata = block_metadata(cost_mlir)
            scheduled_work, schedule = schedule_work(
                eval_work(exprs, config),
                program_count=programs,
                num_ctas=metadata.num_ctas,
                num_sms=num_sms,
            )
            predicted, bottleneck, pipeline_ms = predict_ms(
                scheduled_work, rates, launch_overhead_ms
            )
        except Exception as error:
            skips.append(AnalysisSkip(kernel, ttgir, str(error).splitlines()[0]))
            continue

        rows.append(
            CostResult(
                kernel=kernel,
                ttgir=ttgir,
                time_ms=float(record["time_ms"]),
                predicted_ms=predicted,
                bottleneck=bottleneck,
                scheduled_work=scheduled_work,
                pipeline_ms=pipeline_ms,
                expression=str(sympy.Max(*exprs.values())),
                category_expressions={name: str(expr) for name, expr in exprs.items()},
                args=[str(arg) for arg in record.get("args", [])],
                kwargs={
                    str(key): str(value)
                    for key, value in record.get("kwargs", {}).items()
                },
                grid_size=grid_size,
                schedule=schedule,
            )
        )
    return rows, skips

