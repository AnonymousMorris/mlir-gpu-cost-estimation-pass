from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import Dataset, Sample, estimate


def write_plots(dataset: Dataset, values: dict[str, float], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = [sample for rows in dataset.kernels.values() for sample in rows]
    written: list[Path] = []
    if not samples:
        return written

    written.append(_plot_actual_vs_estimated(samples, values, output_dir / "actual_vs_estimated.png"))
    written.append(_plot_actual_vs_estimated_normalized(samples, values, output_dir / "actual_vs_estimated_normalized.png"))
    written.append(_plot_estimate_ratio(samples, values, output_dir / "estimate_ratio.png"))
    written.append(_plot_residuals(samples, values, output_dir / "residuals.png"))
    if values:
        written.append(_plot_variable_values(values, output_dir / "fitted_variables.png"))

    for kernel_name, kernel_samples in dataset.kernels.items():
        if kernel_samples:
            filename = f"{_slug(kernel_name)}_runtime.png"
            written.append(_plot_kernel_runtime(kernel_name, kernel_samples, values, output_dir / filename))

    return written


def _plot_actual_vs_estimated(samples: list[Sample], values: dict[str, float], path: Path) -> Path:
    actual = np.array([sample.time_ms for sample in samples], dtype=float)
    predicted = np.array([estimate(sample, values) for sample in samples], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for kernel in sorted({sample.kernel for sample in samples}):
        indices = [index for index, sample in enumerate(samples) if sample.kernel == kernel]
        ax.scatter(actual[indices], predicted[indices], label=kernel, alpha=0.8, s=42)

    lower = min(float(actual.min()), float(predicted.min()))
    upper = max(float(actual.max()), float(predicted.max()))
    pad = max((upper - lower) * 0.06, 1e-9)
    ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color="black", linewidth=1, alpha=0.65)
    ax.set_xlim(lower - pad, upper + pad)
    ax.set_ylim(lower - pad, upper + pad)
    ax.set_title("Actual vs Estimated Runtime (Absolute)")
    ax.set_xlabel("Actual runtime (ms)")
    ax.set_ylabel("Estimated runtime (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_actual_vs_estimated_normalized(samples: list[Sample], values: dict[str, float], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for kernel in sorted({sample.kernel for sample in samples}):
        kernel_samples = [sample for sample in samples if sample.kernel == kernel]
        actual = np.array([sample.time_ms for sample in kernel_samples], dtype=float)
        predicted = np.array([estimate(sample, values) for sample in kernel_samples], dtype=float)
        ax.scatter(_normalize(actual), _normalize(predicted), label=kernel, alpha=0.8, s=42)

    ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.65)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Actual vs Estimated Runtime (Normalized per Kernel)")
    ax.set_xlabel("Actual runtime, normalized within kernel")
    ax.set_ylabel("Estimated runtime, normalized within kernel")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_estimate_ratio(samples: list[Sample], values: dict[str, float], path: Path) -> Path:
    actual = np.array([sample.time_ms for sample in samples], dtype=float)
    predicted = np.array([estimate(sample, values) for sample in samples], dtype=float)
    ratio = np.divide(predicted, actual, out=np.full_like(predicted, np.nan), where=actual != 0)
    x = np.arange(len(samples))

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for kernel in sorted({sample.kernel for sample in samples}):
        indices = np.array([index for index, sample in enumerate(samples) if sample.kernel == kernel], dtype=int)
        ax.scatter(x[indices], ratio[indices], label=kernel, alpha=0.8, s=36)

    ax.axhline(1, color="black", linewidth=1)
    finite = ratio[np.isfinite(ratio)]
    if finite.size and np.all(finite > 0):
        ax.set_yscale("log")
        lower, upper = np.percentile(finite, [2, 98])
        ax.set_ylim(max(lower * 0.5, np.finfo(float).tiny), upper * 2.0)
    else:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Estimated / Actual Runtime")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Estimated divided by actual")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_residuals(samples: list[Sample], values: dict[str, float], path: Path) -> Path:
    residuals = np.array([estimate(sample, values) - sample.time_ms for sample in samples], dtype=float)
    x = np.arange(len(samples))

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(x, residuals, color=np.where(residuals >= 0, "#c2410c", "#2563eb"), alpha=0.8)
    ax.set_title("Residuals by Sample")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Estimated - actual runtime (ms)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_variable_values(values: dict[str, float], path: Path) -> Path:
    names = list(values)
    fitted = np.array([values[name] for name in names], dtype=float)
    height = max(4, 0.35 * len(names))

    fig, ax = plt.subplots(figsize=(9, height), constrained_layout=True)
    ax.barh(names, fitted, color="#0f766e", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Fitted Cost Variable Values")
    ax.set_xlabel("Value")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_kernel_runtime(
    kernel_name: str,
    samples: list[Sample],
    values: dict[str, float],
    path: Path,
) -> Path:
    x = np.arange(len(samples))
    actual = np.array([sample.time_ms for sample in samples], dtype=float)
    predicted = np.array([estimate(sample, values) for sample in samples], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(x, _normalize(actual), marker="o", linewidth=1.5, label=f"Actual ({_range_label(actual)} ms)")
    ax.plot(x, _normalize(predicted), marker="x", linewidth=1.5, label=f"Estimated ({_range_label(predicted)} ms)")
    ax.set_title(f"{kernel_name} Runtime Shape")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Normalized within each series")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)

    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    if span > 0:
        return (values - lower) / span

    scale = max(abs(upper), 1.0)
    return values / scale


def _range_label(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "n/a"
    return f"{_compact(float(finite.min()))}-{_compact(float(finite.max()))}"


def _compact(value: float) -> str:
    absolute = abs(value)
    if absolute != 0 and (absolute < 0.001 or absolute >= 10000):
        return f"{value:.2e}"
    return f"{value:.4g}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "kernel"
