from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
COST_MLIR = """func.func @__cost_expr() -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "l1"},
  f64 {cost.name = "memory"}
) attributes {
  cost.num_ctas = 1 : i64,
  cost.threads_per_block = 128 : i64,
  cost.work_unit = "block"
} {
  %zero = arith.constant 0.000000e+00 : f64
  %memory = arith.constant 1.280000e+02 : f64
  return %zero, %zero, %zero, %zero, %zero, %memory : f64, f64, f64, f64, f64, f64
}"""


def test_runs_dataset_analysis_end_to_end(tmp_path: Path) -> None:
    ttgir_dir = tmp_path / "ttgir"
    ttgir_dir.mkdir()
    (ttgir_dir / "sample.ttgir").write_text("module {}\n")

    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "add_kernel": [
                    {
                        "status": "ok",
                        "ttgir_filename": "sample.ttgir",
                        "compiled_name": "add_kernel",
                        "grid_size": [4],
                        "time_ms": 0.01,
                        "args": [],
                        "kwargs": {},
                    }
                ]
            }
        )
    )
    triton_opt = tmp_path / "triton-opt"
    triton_opt.write_text(
        "#!/bin/sh\ncat <<'MLIR'\n" + COST_MLIR + "\nMLIR\n"
    )
    triton_opt.chmod(0o755)

    output_dir = tmp_path / "output"
    output = output_dir / "cost_predictions.json"
    scatter = output_dir / "cost_prediction_scatter.png"
    pipeline_plot = output_dir / "cost_pipeline_counts.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(MAIN),
            str(results),
            str(ttgir_dir),
            "--output",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=True,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
    )

    payload = json.loads(output.read_text())
    prediction = payload["predictions"][0]
    assert completed.stdout == f"wrote 1 predictions to {output}\n"
    assert payload["summary"]["count"] == 1
    assert payload["summary"]["skip_count"] == 0
    assert prediction["kernel"] == "add_kernel"
    assert prediction["bottleneck"] == "memory"
    assert prediction["scheduled_work"]["memory"] == 128.0
    assert prediction["predicted_ms"] > 0
    assert scatter.is_file()
    assert pipeline_plot.is_file()
