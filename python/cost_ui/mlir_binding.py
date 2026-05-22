from __future__ import annotations

import os
import sys
from pathlib import Path


class MlirBindingError(ImportError):
    pass


def load_mlir_ir():
    """Load the LLVM MLIR Python binding, honoring local path configuration."""

    _extend_sys_path_from_env()
    try:
        from mlir.ir import Context, Module
    except ImportError as exc:
        raise MlirBindingError(_diagnostic()) from exc
    return Context, Module


def _extend_sys_path_from_env() -> None:
    configured_path = os.environ.get("MLIR_PYTHON_PACKAGE_DIR")
    if configured_path:
        _prepend_paths(configured_path.split(os.pathsep))
        return

    _prepend_paths(str(path) for path in _candidate_binding_dirs())


def _prepend_paths(paths) -> None:
    for raw_path in paths:
        path = raw_path.strip()
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _diagnostic() -> str:
    candidates = _candidate_binding_dirs()
    candidate_text = "\n".join(f"  - {path}" for path in candidates)
    if not candidate_text:
        candidate_text = "  - none found"

    return (
        "LLVM MLIR Python bindings are not importable as `mlir.ir` in this uv "
        "environment.\n\n"
        "Checked configuration:\n"
        f"  - MLIR_PYTHON_PACKAGE_DIR={os.environ.get('MLIR_PYTHON_PACKAGE_DIR', '<unset>')}\n"
        f"  - sys.executable={sys.executable}\n"
        "Candidate binding directories found on disk:\n"
        f"{candidate_text}\n\n"
        "Your /usr/local MLIR install reports MLIR_ENABLE_BINDINGS_PYTHON=0, "
        "so it does not contain the Python package. Rebuild/install MLIR with "
        "`-DMLIR_ENABLE_BINDINGS_PYTHON=ON`, then run this app with "
        "`MLIR_PYTHON_PACKAGE_DIR=/path/to/build/tools/mlir/python_packages/mlir_core "
        "uv run python main.py`."
    )


def _candidate_binding_dirs() -> list[Path]:
    roots = [
        Path("/home/morris/llvm-project/build/tools/mlir/python_packages/mlir_core"),
        Path("/usr/local/python_packages/mlir_core"),
        Path("/usr/local/tools/mlir/python_packages/mlir_core"),
        Path("../build/tools/mlir/python_packages/mlir_core").resolve(),
        Path("../../llvm-project/build/tools/mlir/python_packages/mlir_core").resolve(),
    ]
    return [path for path in roots if (path / "mlir" / "ir.py").exists()]
