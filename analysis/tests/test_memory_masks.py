from __future__ import annotations

import json

import pytest

from src import analyzer
from src.cost_model import WORK_CATEGORIES
from src.parse import parse_cost_analysis
from src.results import write_output


def cost_mlir(mask_attributes: str = "") -> str:
    results = ", ".join(f'f64 {{cost.name = "{name}"}}' for name in WORK_CATEGORIES)
    zeros = ", ".join("%zero" for _ in WORK_CATEGORIES)
    types = ", ".join("f64" for _ in WORK_CATEGORIES)
    return f"""func.func @__cost_expr() -> ({results}) attributes {{
      cost.num_ctas = 1 : i64, cost.threads_per_block = 32 : i64,
      cost.work_unit = "block" {mask_attributes}
    }} {{
      %zero = arith.constant 0.0 : f64
      return {zeros} : {types}
    }}"""


@pytest.mark.parametrize(("attributes", "model", "notes"), [
    ("", "unknown", ()),
    (', cost.memory_mask_model = "exact"', "exact", ()),
    (', cost.memory_mask_model = "upper_bound", '
     'cost.memory_mask_approximations = ["tt.load: unsupported mask"]',
     "upper_bound", ("tt.load: unsupported mask",)),
])
def test_parses_mask_precision_metadata(attributes, model, notes):
    parsed = parse_cost_analysis(cost_mlir(attributes))
    assert parsed.memory_mask_model == model
    assert parsed.memory_mask_approximations == notes


@pytest.mark.parametrize("attributes", [
    ', cost.memory_mask_model = "guess"',
    ', cost.memory_mask_model = "upper_bound"',
    ', cost.memory_mask_model = "exact", cost.memory_mask_approximations = ["reason"]',
    ', cost.memory_mask_approximations = ["reason"]',
    ', cost.memory_mask_model = "upper_bound", cost.memory_mask_approximations = [""]',
    ', cost.memory_mask_model = "upper_bound", cost.memory_mask_approximations = [1 : i64]',
])
def test_rejects_inconsistent_mask_metadata(attributes):
    with pytest.raises(ValueError, match="memory mask"):
        parse_cost_analysis(cost_mlir(attributes))


def test_propagates_mask_approximation_reasons_to_prediction_json(tmp_path, monkeypatch):
    fixture = tmp_path / "kernel.ttgir"
    fixture.write_text("module {}\n")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"kernel": [{
        "status": "ok", "ttgir_filename": fixture.name,
        "grid_size": [1], "time_ms": 0.01,
    }]}))
    mlir = cost_mlir(
        ', cost.memory_mask_model = "upper_bound", '
        'cost.memory_mask_approximations = ["tt.load: unsupported mask"]'
    )
    monkeypatch.setattr(analyzer, "run_cost_pass", lambda *args: mlir)
    rows, skips = analyzer.analyze(results, tmp_path)
    assert not skips
    output = tmp_path / "predictions.json"
    write_output(output, rows, skips)
    payload = json.loads(output.read_text())
    assert payload["predictions"][0]["memory_mask_model"] == "upper_bound"
    assert payload["predictions"][0]["memory_mask_approximations"] == [
        "tt.load: unsupported mask"
    ]
    assert payload["summary"]["memory_mask_counts"] == {
        "exact": 0, "upper_bound": 1, "unknown": 0
    }
