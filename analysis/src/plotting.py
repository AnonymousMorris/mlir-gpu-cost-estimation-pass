from __future__ import annotations

from pathlib import Path

from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from .cost_model import RESOURCES, RESOURCE_GROUPS
from .results import CostResult


BOUND_MARKERS = {"compute": "o", "memory": "^"}
RESOURCE_LABELS = {"l1": "shared memory", "memory": "global memory"}
PLOT_EXCLUDED_KERNELS = frozenset({"matmul_kernel_persistent"})


def plot(rows: list[CostResult], scatter_path: Path, pipeline_path: Path) -> None:
    rows = plottable_rows(rows)
    if not rows:
        return

    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    colors = kernel_color_map(rows)
    _plot_runtime_scatter(rows, colors, scatter_path)
    _plot_pipeline_counts(rows, colors, pipeline_path)


def plottable_rows(rows: list[CostResult]) -> list[CostResult]:
    return [row for row in rows if row.kernel not in PLOT_EXCLUDED_KERNELS]


def kernel_color_map(rows: list[CostResult]) -> dict[str, tuple[float, ...]]:
    kernels = sorted({row.kernel for row in rows})
    if len(kernels) <= 10:
        palette = colormaps["tab10"].colors
    elif len(kernels) <= 20:
        palette = colormaps["tab20"].colors
    else:
        color_map = colormaps["hsv"].resampled(len(kernels))
        palette = [color_map(index) for index in range(len(kernels))]
    return {
        kernel: tuple(float(channel) for channel in palette[index])
        for index, kernel in enumerate(kernels)
    }


def _plot_runtime_scatter(
    rows: list[CostResult],
    colors: dict[str, tuple[float, ...]],
    path: Path,
) -> None:
    kernels = list(colors)
    xs = [row.time_ms for row in rows]
    ys = [row.predicted_ms for row in rows]
    min_axis = min(xs + ys)
    max_axis = max(xs + ys)
    guide_range = [min_axis / 1.25, max_axis * 1.25]

    fig, ax = plt.subplots(figsize=(10, 7))
    for kernel in kernels:
        for bound in RESOURCE_GROUPS:
            group = [
                row
                for row in rows
                if row.kernel == kernel and row.bound == bound
            ]
            if group:
                ax.scatter(
                    [row.time_ms for row in group],
                    [row.predicted_ms for row in group],
                    s=30,
                    alpha=0.78,
                    color=colors[kernel],
                    marker=BOUND_MARKERS[bound],
                    edgecolors="white",
                    linewidths=0.35,
                )
    ax.plot(guide_range, guide_range, color="#222222", linewidth=1.1)
    ax.plot(
        guide_range,
        [value * 2 for value in guide_range],
        color="#777777",
        linewidth=0.8,
        linestyle="--",
    )
    ax.plot(
        guide_range,
        [value / 2 for value in guide_range],
        color="#777777",
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(guide_range)
    ax.set_ylim(guide_range)
    ax.set_box_aspect(1)
    ax.set_title("Measured vs predicted runtime")
    ax.set_xlabel("measured runtime (ms)")
    ax.set_ylabel("predicted runtime (ms)")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)

    kernel_legend = ax.legend(
        handles=_kernel_legend_handles(colors),
        title="kernel",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize="small",
    )
    ax.add_artist(kernel_legend)
    ax.legend(
        handles=_bound_legend_handles(rows),
        title="predicted bound",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0,
        fontsize="small",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _kernel_legend_handles(
    colors: dict[str, tuple[float, ...]],
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            label=kernel,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=7,
        )
        for kernel, color in colors.items()
    ]


def _bound_legend_handles(rows: list[CostResult]) -> list[Line2D]:
    present_bounds = [
        bound
        for bound in RESOURCE_GROUPS
        if any(row.bound == bound for row in rows)
    ]
    return [
        Line2D(
            [0],
            [0],
            marker=BOUND_MARKERS[bound],
            linestyle="none",
            label=f"{bound}-bound",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=7,
        )
        for bound in present_bounds
    ]


def _plot_pipeline_counts(
    rows: list[CostResult],
    colors: dict[str, tuple[float, ...]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 3]}
    )
    for ax, categories, attribute, title in (
        (axes[0], tuple(RESOURCE_GROUPS), "bound", "Compute vs memory"),
        (axes[1], RESOURCES, "bottleneck", "Limiting resources"),
    ):
        bottoms = [0] * len(categories)
        labels = [
            RESOURCE_LABELS.get(category, category) if attribute == "bottleneck" else category
            for category in categories
        ]
        for kernel, color in colors.items():
            counts = [
                sum(
                    row.kernel == kernel and getattr(row, attribute) == category
                    for row in rows
                )
                for category in categories
            ]
            ax.bar(labels, counts, bottom=bottoms, color=color, label=kernel)
            bottoms = [bottom + count for bottom, count in zip(bottoms, counts)]
        for index, total in enumerate(bottoms):
            if total:
                ax.text(index, total, str(total), ha="center", va="bottom")
        ax.set_ylim(0, max(1, max(bottoms)) * 1.12)
        ax.set_title(title)
        ax.set_ylabel("configuration count")
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(
        handles=[
            Patch(facecolor=color, label=kernel)
            for kernel, color in colors.items()
        ],
        title="kernel",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize="small",
    )
    fig.suptitle("Predicted bottlenecks by kernel")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
