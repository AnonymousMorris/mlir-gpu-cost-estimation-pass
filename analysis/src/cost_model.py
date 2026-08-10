from __future__ import annotations

import math
from typing import Any

import sympy


PIPELINES = ("fp32", "fp64", "sfu", "tensor", "memory")


def pipeline_exprs(
    equations: dict[str, sympy.Expr],
) -> dict[str, sympy.Expr]:
    unknown = equations.keys() - set(PIPELINES)
    missing = set(PIPELINES) - equations.keys()
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown categories: {', '.join(sorted(unknown))}")
        if missing:
            details.append(f"missing categories: {', '.join(sorted(missing))}")
        raise ValueError("invalid cost categories (" + "; ".join(details) + ")")
    return {pipeline: equations[pipeline] for pipeline in PIPELINES}


def grid_program_count(grid_size: list[int]) -> int:
    if not grid_size:
        raise ValueError("missing grid_size")
    if any(dimension <= 0 for dimension in grid_size):
        raise ValueError(f"grid_size dimensions must be positive: {grid_size}")
    return math.prod(grid_size)


def eval_work(
    exprs: dict[str, sympy.Expr],
    config: dict[str, Any],
) -> dict[str, float]:
    substitutions = {
        sympy.Symbol(name): float(spec.get("ops_per_count", 1.0))
        for name, spec in config.get("parameters", {}).items()
    }
    default_weight = float(config.get("defaults", {}).get("ops_per_count", 1.0))
    work: dict[str, float] = {}
    for pipeline, expr in exprs.items():
        missing = expr.free_symbols - substitutions.keys()
        local_subs = substitutions | {symbol: default_weight for symbol in missing}
        work[pipeline] = float(expr.evalf(subs=local_subs))
    return work


def per_sm_throughput_rates(
    spec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    sms = int(spec["sms"])
    if sms <= 0:
        raise ValueError("sms must be positive")
    clock_ghz = float(spec["assumed_clock_ghz"])
    per_sm = spec.get("per_sm_ops_per_cycle", {})
    utilization = spec.get("utilization", {})

    def ops_rate(key: str, util_key: str | None = None) -> float:
        util = float(utilization.get(util_key or key, 1.0))
        return clock_ghz * 1_000_000.0 * float(per_sm[key]) * util

    tensor_key = config.get("tensor_throughput_key", "tensor_tf32")
    fp64_ops = per_sm.get("fp64")
    if fp64_ops is None:
        fallback_ratio = float(
            config.get("fp64_fallback_ratio_vs_fp32", 1.0 / 64.0)
        )
        fp64_ops = float(per_sm["fp32"]) * fallback_ratio
    fp64_util = float(utilization.get("fp64", utilization.get("fp32", 1.0)))
    memory_util = float(utilization.get("memory", 1.0))
    per_sm_memory_bandwidth = (
        float(spec["memory_bandwidth_gb_s"]) * 1_000_000.0 * memory_util / sms
    )

    return {
        "fp32": ops_rate("fp32"),
        "fp64": clock_ghz * 1_000_000.0 * float(fp64_ops) * fp64_util,
        "sfu": ops_rate("sfu"),
        "tensor": ops_rate(tensor_key, tensor_key),
        "memory": per_sm_memory_bandwidth,
    }


def predict_ms(
    work: dict[str, float],
    rates: dict[str, float],
    launch_overhead_ms: float,
) -> tuple[float, str, dict[str, float]]:
    pipeline_ms = {
        pipeline: (work[pipeline] / rates[pipeline] if rates[pipeline] else math.inf)
        for pipeline in PIPELINES
    }
    bottleneck = max(pipeline_ms, key=pipeline_ms.get)
    return launch_overhead_ms + pipeline_ms[bottleneck], bottleneck, pipeline_ms
