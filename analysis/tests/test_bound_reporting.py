from __future__ import annotations

import json
from pathlib import Path

from matplotlib.figure import Figure
import pytest

from src.cost_model import WORK_CATEGORIES, RESOURCES, grouped_times, predict_ms
from src.plotting import plot
from src.results import CostResult, summarize, write_output
from src.scheduler import schedule_work


def make_result(work: dict[str, float], *, shared_alu: bool = True) -> CostResult:
    rates = dict.fromkeys(WORK_CATEGORIES, 1.0)
    if shared_alu:
        rates["fp32_int32"] = 1.0
    predicted, bottleneck, times = predict_ms(work, rates, 0.5)
    _, schedule = schedule_work(work, program_count=1, num_ctas=1, num_sms=30)
    return CostResult(
        kernel="example", ttgir="example.ttgir", time_ms=1.0,
        predicted_ms=predicted, bottleneck=bottleneck,
        scheduled_work=work, pipeline_ms=times,
        category_expressions={key: str(value) for key, value in work.items()},
        args=[], kwargs={}, scalar_args={}, grid_size=[1], schedule=schedule,
    )


@pytest.mark.parametrize("resource", RESOURCES)
def test_reports_bound_for_each_limiting_resource(resource):
    work = dict.fromkeys(WORK_CATEGORIES, 0.0)
    if resource == "fp32_int32":
        work.update(fp32=2.0, int32=2.0)
    else:
        work["tensor_fp16" if resource == "tensor" else resource] = 4.0
    row = make_result(work)
    expected_bound = "memory" if resource in ("l1", "memory") else "compute"

    assert row.bottleneck == resource
    assert row.bound == expected_bound
    assert row.group_ms[expected_bound] == 4.0
    assert row.predicted_ms == 0.5 + max(row.group_ms.values())


@pytest.mark.parametrize("shared_alu", [False, True])
def test_groups_times_by_max_not_sum_without_changing_predictions(shared_alu):
    work = dict(fp32=2.0, fp64=3.0, int32=2.0, sfu=1.0,
                tensor_fp16=3.0, tensor_tf32=2.0, l1=6.0, memory=7.0)
    row = make_result(work, shared_alu=shared_alu)

    assert row.group_ms == {"compute": 5.0, "memory": 7.0}
    assert row.predicted_ms == 7.5
    assert row.bound == "memory"
    assert ("fp32_int32" in row.pipeline_ms) == shared_alu


@pytest.mark.parametrize(
    ("work_updates", "bottleneck", "bound"),
    [
        ({}, "fp32", "compute"),
        ({"fp32": 2.0, "memory": 2.0}, "fp32", "compute"),
        ({"fp32": 1.0, "int32": 1.0, "memory": 2.0}, "memory", "memory"),
    ],
)
def test_bound_preserves_existing_resource_tie_breaking(work_updates, bottleneck, bound):
    row = make_result(dict.fromkeys(WORK_CATEGORIES, 0.0) | work_updates)
    assert row.bottleneck == bottleneck
    assert row.bound == bound
    assert row.group_ms["compute"] == row.group_ms["memory"]
    assert row.predicted_ms == 0.5 + max(row.group_ms.values())


def test_resource_groups_cover_every_resource_exactly_once():
    # Each resource alone contributes to one group, never both.
    for resource in RESOURCES:
        times = dict.fromkeys(RESOURCES, 0.0)
        times[resource] = 1.0
        assert sorted(grouped_times(times).values()) == [0.0, 1.0]


def test_serializes_grouped_times_and_preserves_resource_details(tmp_path):
    row = make_result(dict.fromkeys(WORK_CATEGORIES, 0.0) | {"l1": 4.0})
    path = tmp_path / "predictions.json"
    write_output(path, [row], [])
    payload = json.loads(path.read_text())
    prediction = payload["predictions"][0]

    assert prediction["bound"] == "memory"
    assert prediction["group_ms"] == {"compute": 0.0, "memory": 4.0}
    assert prediction["bottleneck"] == "l1"
    assert prediction["pipeline_ms"] == row.pipeline_ms
    assert prediction["predicted_ms"] == row.predicted_ms
    assert payload["summary"]["bound_counts"] == {"compute": 0, "memory": 1}


def test_summary_counts_both_bounds_and_empty_results():
    rows = [
        make_result(dict.fromkeys(WORK_CATEGORIES, 0.0) | {resource: 4.0})
        for resource in ("tensor_fp16", "l1", "memory")
    ]
    assert summarize(rows, [])["bound_counts"] == {"compute": 1, "memory": 2}
    assert summarize([], [])["bound_counts"] == {"compute": 0, "memory": 0}


def test_plots_primary_bounds_and_detailed_resources(tmp_path, monkeypatch):
    figures = {}

    def capture_figure(figure, path, **kwargs):
        figures[Path(path).name] = figure

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    rows = [
        make_result(dict.fromkeys(WORK_CATEGORIES, 0.0) | {resource: 4.0})
        for resource in ("tensor_fp16", "l1", "memory")
    ]
    plot(rows, tmp_path / "scatter.png", tmp_path / "counts.png")

    scatter = figures["scatter.png"].axes[0]
    assert [text.get_text() for text in scatter.get_legend().get_texts()] == [
        "compute-bound", "memory-bound"
    ]
    assert [handle.get_marker() for handle in scatter.get_legend().legend_handles] == [
        "o", "^"
    ]
    assert sum(len(collection.get_offsets()) for collection in scatter.collections) == 3

    primary, detailed = figures["counts.png"].axes
    assert [tick.get_text() for tick in primary.get_xticklabels()] == ["compute", "memory"]
    assert [bar.get_height() for bar in primary.patches] == [1, 2]
    assert primary.get_ylim()[1] > 2
    labels = [tick.get_text() for tick in detailed.get_xticklabels()]
    counts = dict(zip(labels, (bar.get_height() for bar in detailed.patches)))
    assert counts["shared memory"] == 1
    assert counts["global memory"] == 1
    assert counts["tensor"] == 1
    assert "l1" not in labels
