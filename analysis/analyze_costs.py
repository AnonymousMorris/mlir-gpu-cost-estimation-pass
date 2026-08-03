from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import sympy

from mlir_sympy import build_equations, parse_cost_function
from scheduler import Schedule, schedule_work


PIPELINES = ("fp32", "fp64", "sfu", "tensor", "memory")
PLOT_EXCLUDED_KERNELS = frozenset({"matmul_kernel_persistent"})
ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
COST_FUNC_RE = re.compile(r"func\.func\s+@__cost_expr\b")


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


@dataclass(frozen=True)
class BlockMetadata:
    num_ctas: int
    threads_per_block: int


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def iter_records(results: dict[str, list[dict[str, Any]]]):
    for kernel, records in results.items():
        for record in records:
            if record.get("status") == "ok" and record.get("ttgir_filename"):
                yield kernel, record


def pass_pipeline(func_name: str) -> str:
    return f"builtin.module(my-cost-analysis{{func-name={func_name}}})"


def run_cost_pass(
    triton_opt: Path,
    plugin: Path,
    ttgir_path: Path,
    func_name: str,
    timeout_s: float,
) -> str:
    command = [
        str(triton_opt),
        "--load-pass-plugin",
        str(plugin),
        "--pass-pipeline",
        pass_pipeline(func_name),
        str(ttgir_path),
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"triton-opt timed out after {timeout_s:g}s") from error
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(output.strip() or f"triton-opt failed with {result.returncode}")
    return extract_cost_function(output)


def extract_cost_function(text: str) -> str:
    match = COST_FUNC_RE.search(text)
    if not match:
        raise ValueError("pass output did not contain func.func @__cost_expr")

    start = match.start()
    arrow = text.find("->", match.end())
    if arrow < 0:
        raise ValueError("cost function has no return type")
    result_paren_depth = 0
    brace_start = None
    index = arrow + 2
    while index < len(text):
        char = text[index]
        if char == "(":
            result_paren_depth += 1
        elif char == ")":
            result_paren_depth -= 1
        elif char == "{" and result_paren_depth == 0:
            prefix = text[arrow + 2 : index].rstrip()
            if prefix.endswith("attributes"):
                attribute_depth = 1
                while attribute_depth and index + 1 < len(text):
                    index += 1
                    if text[index] == "{":
                        attribute_depth += 1
                    elif text[index] == "}":
                        attribute_depth -= 1
                index += 1
                continue
            brace_start = index
            break
        index += 1
    if brace_start is None:
        raise ValueError("cost function has no body")

    depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("cost function body is unterminated")


def pipeline_exprs(cost_mlir: str) -> dict[str, sympy.Expr]:
    equations = build_equations(parse_cost_function(cost_mlir))
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


def block_metadata(cost_mlir: str) -> BlockMetadata:
    cost_function = parse_cost_function(cost_mlir)
    attributes = cost_function.operation.operation.attributes
    try:
        work_unit = attributes["cost.work_unit"].value
        metadata = BlockMetadata(
            num_ctas=int(attributes["cost.num_ctas"].value),
            threads_per_block=int(attributes["cost.threads_per_block"].value),
        )
    except KeyError as error:
        raise ValueError(f"cost function is missing {error.args[0]}") from error

    if work_unit != "block":
        raise ValueError(f"unsupported cost work unit: {work_unit}")
    if metadata.num_ctas <= 0:
        raise ValueError("cost.num_ctas must be positive")
    if metadata.threads_per_block <= 0:
        raise ValueError("cost.threads_per_block must be positive")
    return metadata


def grid_program_count(grid_size: list[int]) -> int:
    if not grid_size:
        raise ValueError("missing grid_size")
    if any(dimension <= 0 for dimension in grid_size):
        raise ValueError(f"grid_size dimensions must be positive: {grid_size}")
    return math.prod(grid_size)


def eval_work(exprs: dict[str, sympy.Expr], config: dict[str, Any]) -> dict[str, float]:
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
    spec: dict[str, Any], config: dict[str, Any]
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


def analyze(args: argparse.Namespace) -> tuple[list[CostResult], list[AnalysisSkip]]:
    results = load_json(args.results)
    spec = load_json(args.gpu_spec)
    config = load_json(args.config)
    rates = per_sm_throughput_rates(spec, config)
    num_sms = int(spec["sms"])
    launch_overhead_ms = float(spec.get("launch_overhead_ms", 0.0))

    rows: list[CostResult] = []
    skips: list[AnalysisSkip] = []
    for index, (kernel, record) in enumerate(iter_records(results)):
        if args.limit is not None and index >= args.limit:
            break

        ttgir = record["ttgir_filename"]
        ttgir_path = args.ttgir_dir / ttgir
        func_name = record.get("compiled_name") or kernel
        if not ttgir_path.exists():
            skips.append(AnalysisSkip(kernel, ttgir, "missing ttgir"))
            continue

        try:
            grid_size = [int(v) for v in record.get("grid_size", [])]
            programs = grid_program_count(grid_size)
            cost_mlir = run_cost_pass(
                args.triton_opt,
                args.plugin,
                ttgir_path,
                func_name,
                args.timeout,
            )
            exprs = pipeline_exprs(cost_mlir)
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
                kwargs={str(k): str(v) for k, v in record.get("kwargs", {}).items()},
                grid_size=grid_size,
                schedule=schedule,
            )
        )
    return rows, skips


def write_output(path: Path, rows: list[CostResult], skips: list[AnalysisSkip]) -> None:
    payload = {
        "predictions": [asdict(row) for row in rows],
        "analysis_skips": [asdict(skip) for skip in skips],
        "summary": summarize(rows, skips),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize(rows: list[CostResult], skips: list[AnalysisSkip]) -> dict[str, Any]:
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


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def plottable_rows(rows: list[CostResult]) -> list[CostResult]:
    return [row for row in rows if row.kernel not in PLOT_EXCLUDED_KERNELS]


def plot(rows: list[CostResult], scatter_path: Path, pipeline_path: Path) -> None:
    rows = plottable_rows(rows)
    if not rows:
        return

    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "fp32": "#3b82f6",
        "fp64": "#7c3aed",
        "sfu": "#ef4444",
        "tensor": "#10b981",
        "memory": "#f59e0b",
    }
    xs = [row.time_ms for row in rows]
    ys = [row.predicted_ms for row in rows]
    max_axis = max(xs + ys)

    fig, ax = plt.subplots(figsize=(8, 7))
    for pipeline in PIPELINES:
        group = [row for row in rows if row.bottleneck == pipeline]
        if group:
            ax.scatter(
                [row.time_ms for row in group],
                [row.predicted_ms for row in group],
                s=26,
                alpha=0.75,
                label=pipeline,
                color=colors[pipeline],
            )
    ax.plot([0, max_axis], [0, max_axis], color="#222222", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured runtime (ms)")
    ax.set_ylabel("predicted runtime (ms)")
    ax.legend(title="bottleneck")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)

    counts = {pipeline: 0 for pipeline in PIPELINES}
    for row in rows:
        counts[row.bottleneck] += 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(counts), list(counts.values()), color=[colors[p] for p in counts])
    ax.set_xlabel("predicted bottleneck")
    ax.set_ylabel("kernel count")
    fig.tight_layout()
    fig.savefig(pipeline_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cost-analysis pass over TGIR metadata and plot throughput predictions.")
    parser.add_argument("--results", type=Path, default=ROOT / "data" / "result.json")
    parser.add_argument("--ttgir-dir", type=Path, default=ROOT / "data" / "ttgir")
    parser.add_argument("--gpu-spec", type=Path, default=ROOT / "gpu_spec.json")
    parser.add_argument("--config", type=Path, default=ROOT / "cost_analysis_config.json")
    parser.add_argument("--triton-opt", type=Path, default=Path("/home/morris/bin/triton-opt"))
    parser.add_argument("--plugin", type=Path, default=ROOT.parent / "build" / "libMyPass.so")
    parser.add_argument("--output", type=Path, default=ROOT / "cost_predictions.json")
    parser.add_argument("--scatter", type=Path, default=PLOTS_DIR / "cost_prediction_scatter.png")
    parser.add_argument("--pipeline-plot", type=Path, default=PLOTS_DIR / "cost_pipeline_counts.png")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=10.0, help="per-kernel triton-opt timeout in seconds")
    args = parser.parse_args()

    rows, skips = analyze(args)
    write_output(args.output, rows, skips)
    plot(rows, args.scatter, args.pipeline_plot)
    print(f"wrote {len(rows)} predictions to {args.output}")
    if skips:
        print(f"skipped {len(skips)} records")


if __name__ == "__main__":
    main()
