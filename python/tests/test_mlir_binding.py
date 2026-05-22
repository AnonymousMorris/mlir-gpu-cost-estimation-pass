import sys

from cost_ui.mlir_binding import _extend_sys_path_from_env


def test_extends_sys_path_from_mlir_python_package_dir(monkeypatch, tmp_path):
    package_dir = str(tmp_path)
    monkeypatch.setenv("MLIR_PYTHON_PACKAGE_DIR", package_dir)
    monkeypatch.setattr(sys, "path", [])

    _extend_sys_path_from_env()

    assert sys.path == [package_dir]


def test_auto_discovers_candidate_binding_dir(monkeypatch, tmp_path):
    package_dir = tmp_path / "mlir_core"
    (package_dir / "mlir").mkdir(parents=True)
    (package_dir / "mlir" / "ir.py").write_text("")
    monkeypatch.delenv("MLIR_PYTHON_PACKAGE_DIR", raising=False)
    monkeypatch.setattr("cost_ui.mlir_binding._candidate_binding_dirs", lambda: [package_dir])
    monkeypatch.setattr(sys, "path", [])

    _extend_sys_path_from_env()

    assert sys.path == [str(package_dir)]
