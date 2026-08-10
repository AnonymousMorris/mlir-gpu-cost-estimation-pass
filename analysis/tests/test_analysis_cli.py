from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
RUNTIME_COST_MLIR = """func.func @__cost_expr(
  %K: i32 {cost.kind = "runtime", cost.name = "K"}
) -> (
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
  %base = arith.constant 2.097152e+06 : f64
  %tile = arith.constant 64 : i32
  %iterations = arith.ceildivui %K, %tile : i32
  %iterations_f = arith.uitofp %iterations : i32 to f64
  %tensor = arith.mulf %base, %iterations_f : f64
  return %zero, %zero, %zero, %tensor, %zero, %zero : f64, f64, f64, f64, f64, f64
}"""


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


def run_analysis(
    tmp_path: Path,
    cost_mlir: str,
    results_payload: dict,
    output_name: str = "output",
) -> tuple[subprocess.CompletedProcess[str], dict, Path]:
    ttgir_dir = tmp_path / "ttgir"
    ttgir_dir.mkdir(exist_ok=True)
    for records in results_payload.values():
        for record in records:
            (ttgir_dir / record["ttgir_filename"]).write_text("module {}\n")

    results = tmp_path / f"{output_name}-results.json"
    results.write_text(json.dumps(results_payload))
    triton_opt = tmp_path / "triton-opt"
    triton_opt.write_text(
        "#!/bin/sh\ncat <<'MLIR'\n" + cost_mlir + "\nMLIR\n"
    )
    triton_opt.chmod(0o755)

    output_dir = tmp_path / output_name
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
    output = output_dir / "cost_predictions.json"
    return completed, json.loads(output.read_text()), output_dir


def test_runs_dataset_analysis_end_to_end(tmp_path: Path) -> None:
    completed, payload, output_dir = run_analysis(
        tmp_path,
        COST_MLIR,
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
        },
    )

    output = output_dir / "cost_predictions.json"
    prediction = payload["predictions"][0]
    assert completed.stdout == f"wrote 1 predictions to {output}\n"
    assert payload["summary"]["count"] == 1
    assert payload["summary"]["skip_count"] == 0
    assert prediction["kernel"] == "add_kernel"
    assert prediction["bottleneck"] == "memory"
    assert prediction["scheduled_work"]["memory"] == 128.0
    assert prediction["predicted_ms"] > 0
    assert (output_dir / "cost_prediction_scatter.png").is_file()
    assert (output_dir / "cost_pipeline_counts.png").is_file()


def test_binds_runtime_arguments_in_symbolic_costs(tmp_path: Path) -> None:
    record = {
        "status": "ok",
        "ttgir_filename": "matmul.ttgir",
        "compiled_name": "matmul_kernel",
        "grid_size": [1],
        "time_ms": 0.01,
        "args": ["256"],
        "kwargs": {},
        "scalar_args": {"K": 256},
    }
    _, payload, _ = run_analysis(
        tmp_path,
        RUNTIME_COST_MLIR,
        {"matmul_kernel": [record]},
    )

    prediction = payload["predictions"][0]
    assert prediction["scheduled_work"]["tensor"] == 8_388_608.0

    del record["scalar_args"]
    _, missing_payload, _ = run_analysis(
        tmp_path,
        RUNTIME_COST_MLIR,
        {"matmul_kernel": [record]},
        output_name="missing-output",
    )
    assert missing_payload["predictions"] == []
    assert missing_payload["analysis_skips"][0]["reason"] == (
        "missing runtime argument bindings: K"
    )
