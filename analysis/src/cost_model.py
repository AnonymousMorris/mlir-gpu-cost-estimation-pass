from __future__ import annotations

import math
from typing import Any

import sympy


PIPELINES = ("fp32", "fp64", "sfu", "tensor", "l1", "memory")


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
    runtime_args: dict[str, Any],
    *,
    weight_symbols: frozenset[str],
    runtime_symbols: frozenset[str],
) -> dict[str, float]:
    overlap = weight_symbols & runtime_symbols
    if overlap:
        raise ValueError(
            "cost symbols have conflicting roles: " + ", ".join(sorted(overlap))
        )

    free_symbols = set().union(*(expr.free_symbols for expr in exprs.values()))
    declared_symbols = {
        sympy.Symbol(name) for name in weight_symbols | runtime_symbols
    }
    unclassified = free_symbols - declared_symbols
    if unclassified:
        raise ValueError(
            "unclassified cost symbols: "
            + ", ".join(sorted(str(symbol) for symbol in unclassified))
        )

    required_runtime = {
        str(symbol) for symbol in free_symbols
        if str(symbol) in runtime_symbols
    }
    missing_runtime = required_runtime - runtime_args.keys()
    if missing_runtime:
        raise ValueError(
            "missing runtime argument bindings: "
            + ", ".join(sorted(missing_runtime))
        )

    runtime_substitutions: dict[sympy.Symbol, sympy.Expr] = {}
    for name in required_runtime:
        value = runtime_args[name]
        if isinstance(value, bool):
            runtime_substitutions[sympy.Symbol(name)] = sympy.Integer(int(value))
        elif isinstance(value, int):
            runtime_substitutions[sympy.Symbol(name)] = sympy.Integer(value)
        elif isinstance(value, float) and math.isfinite(value):
            runtime_substitutions[sympy.Symbol(name)] = sympy.Float(value)
        else:
            raise ValueError(
                f"runtime argument {name} must be a finite numeric value"
            )

    configured_weights = config.get("parameters", {})
    default_weight = float(config.get("defaults", {}).get("ops_per_count", 1.0))
    weight_substitutions = {
        sympy.Symbol(name): float(
            configured_weights.get(name, {}).get("ops_per_count", default_weight)
        )
        for name in weight_symbols
    }
    substitutions = weight_substitutions | runtime_substitutions
    return {
        pipeline: float(expr.evalf(subs=substitutions))
        for pipeline, expr in exprs.items()
    }


def per_sm_throughput_rates(
    spec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    sms = int(spec["sms"])
    if sms <= 0:
        raise ValueError("sms must be positive")
    clock_ghz = float(spec["assumed_clock_ghz"])
    per_sm_ops = spec.get("per_sm_ops_per_cycle", {})
    per_sm_bytes = spec.get("per_sm_bytes_per_cycle", {})
    utilization = spec.get("utilization", {})

    def ops_rate(key: str, util_key: str | None = None) -> float:
        util = float(utilization.get(util_key or key, 1.0))
        return clock_ghz * 1_000_000.0 * float(per_sm_ops[key]) * util

    def bytes_rate(key: str) -> float:
        util = float(utilization.get(key, 1.0))
        return clock_ghz * 1_000_000.0 * float(per_sm_bytes[key]) * util

    tensor_key = config.get("tensor_throughput_key", "tensor_tf32")
    fp64_ops = per_sm_ops.get("fp64")
    if fp64_ops is None:
        fallback_ratio = float(
            config.get("fp64_fallback_ratio_vs_fp32", 1.0 / 64.0)
        )
        fp64_ops = float(per_sm_ops["fp32"]) * fallback_ratio
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
        "l1": bytes_rate("l1"),
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
