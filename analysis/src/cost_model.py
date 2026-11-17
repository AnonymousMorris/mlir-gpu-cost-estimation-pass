from __future__ import annotations

import math
from typing import Any

import sympy


# Scalar categories use per-lane instruction-equivalents; tensor uses FLOPs;
# l1 and memory use bytes. The shared resource has no separate work equation.
PIPELINES = ("fp32", "fp64", "int32", "sfu", "tensor", "l1", "memory")
RESOURCES = (*PIPELINES, "fp32_int32")


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
    required_weights = {str(symbol) for symbol in free_symbols} & weight_symbols
    weight_substitutions = {}
    for name in sorted(required_weights):
        parameter = configured_weights.get(name, {})
        if "work_per_count" not in parameter:
            raise ValueError(f"missing work_per_count for cost weight: {name}")
        if parameter["work_per_count"] is None:
            reason = parameter.get("unmodeled_reason", "no cost model or calibrated weight")
            raise ValueError(
                f"unmodeled cost weight {name}: {reason} "
                "Set an explicit work_per_count to use a calibrated approximation."
            )
        weight = float(parameter["work_per_count"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"cost weight {name} must be finite and nonnegative")
        weight_substitutions[sympy.Symbol(name)] = weight
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
    utilization = spec.get("utilization", {})
    cycles_per_ms = clock_ghz * 1_000_000.0

    def rate(capacity: float, key: str) -> float:
        result = float(capacity) * float(utilization.get(key, 1.0))
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"effective capacity for {key} must be finite and positive")
        return result

    if not math.isfinite(cycles_per_ms) or cycles_per_ms <= 0:
        raise ValueError("assumed_clock_ghz must be finite and positive")

    instructions = spec["per_sm_instructions_per_cycle"]
    tensor_key = config.get("tensor_throughput_key", "tensor_tf32")
    rates = {
        key: cycles_per_ms * rate(instructions[key], key)
        for key in ("fp32", "fp64", "int32", "sfu")
    }
    rates["tensor"] = cycles_per_ms * rate(
        spec["per_sm_flops_per_cycle"][tensor_key], tensor_key
    )
    rates["l1"] = cycles_per_ms * rate(spec["per_sm_bytes_per_cycle"]["l1"], "l1")
    rates["memory"] = rate(spec["memory_bandwidth_gb_s"], "memory") * 1_000_000 / sms
    # Omit this capacity for architectures without GA10x-style sharing.
    if "fp32_int32" in instructions:
        rates["fp32_int32"] = cycles_per_ms * rate(
            instructions["fp32_int32"], "fp32_int32"
        )
    return rates


def predict_ms(
    work: dict[str, float],
    rates: dict[str, float],
    launch_overhead_ms: float,
) -> tuple[float, str, dict[str, float]]:
    pipeline_ms = {
        pipeline: (work[pipeline] / rates[pipeline] if rates[pipeline] else math.inf)
        for pipeline in PIPELINES
    }
    if "fp32_int32" in rates:
        pipeline_ms["fp32_int32"] = (
            work["fp32"] + work["int32"]
        ) / rates["fp32_int32"]
    bottleneck = max(pipeline_ms, key=pipeline_ms.get)
    return launch_overhead_ms + pipeline_ms[bottleneck], bottleneck, pipeline_ms
