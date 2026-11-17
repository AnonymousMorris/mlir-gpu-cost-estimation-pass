from dataclasses import dataclass
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class OutputPaths:
    """Sweep artifacts under one root, relative to the working directory by default."""

    root: Path = Path("results")

    @property
    def result_path(self) -> Path:
        return self.root / "result.json"

    @property
    def resume_path(self) -> Path:
        return self.root / "run_state.json.tmp"

    @property
    def ttgir_dir(self) -> Path:
        return self.root / "ttgir"


RESUME_VERSION = 2


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = path.with_name(f"{path.name}.new")
    with pending_path.open("w") as file:
        json.dump(value, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(pending_path, path)


def write_result(result, *, paths: OutputPaths = OutputPaths()):
    atomic_write_json(paths.result_path, result)


def save_resume_state(
    result,
    kernel_names,
    warmup_ms,
    rep_ms,
    next_kernel_index,
    next_case_index,
    *,
    paths: OutputPaths = OutputPaths(),
):
    atomic_write_json(
        paths.resume_path,
        {
            "version": RESUME_VERSION,
            "kernel_names": kernel_names,
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            "next_kernel_index": next_kernel_index,
            "next_case_index": next_case_index,
            "result": result,
        },
    )


def load_resume_state(
    kernel_names, warmup_ms, rep_ms, *, paths: OutputPaths = OutputPaths()
):
    if not paths.resume_path.exists():
        return {
            "next_kernel_index": 0,
            "next_case_index": 0,
            "result": {},
        }

    try:
        state = json.loads(paths.resume_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read {paths.resume_path}; use --restart to discard it"
        ) from exc

    expected = {
        "version": RESUME_VERSION,
        "kernel_names": kernel_names,
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
    }
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        fields = ", ".join(mismatches)
        raise RuntimeError(
            f"Resume state does not match this run ({fields}); "
            "use --restart to start over"
        )

    next_kernel_index = state.get("next_kernel_index")
    next_case_index = state.get("next_case_index")
    result = state.get("result")
    valid_cursor = (
        isinstance(next_kernel_index, int)
        and 0 <= next_kernel_index <= len(kernel_names)
        and isinstance(next_case_index, int)
        and next_case_index >= 0
        and (next_kernel_index < len(kernel_names) or next_case_index == 0)
    )
    if not valid_cursor or not isinstance(result, dict):
        raise RuntimeError(
            f"Invalid resume state in {paths.resume_path}; use --restart to discard it"
        )

    return state


def write_ttgir(name, ttgir, *, paths: OutputPaths = OutputPaths()):
    paths.ttgir_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)
    filename = f"{safe_name}.ttgir"
    (paths.ttgir_dir / filename).write_text(ttgir)
    return filename
