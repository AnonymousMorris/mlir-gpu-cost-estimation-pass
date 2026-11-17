from pathlib import Path
import subprocess
import sys

import pytest

import main


MAIN_PATH = Path(main.__file__).resolve()


def test_default_options():
    args = main.parse_args([])

    assert args.warmup_ms == 100
    assert args.rep_ms == 1000
    assert args.restart is False


def test_zero_warmup_is_allowed():
    assert main.parse_args(["--warmup-ms", "0"]).warmup_ms == 0


@pytest.mark.parametrize(
    "option, value",
    [
        ("--warmup-ms", "-1"),
        ("--rep-ms", "-1"),
        ("--rep-ms", "0"),
        ("--warmup-ms", "abc"),
        ("--rep-ms", "1.5"),
    ],
)
def test_invalid_durations_are_rejected_before_gpu_imports(option, value):
    # -S removes site-packages, so importing Torch or Triton would fail here.
    result = subprocess.run(
        [sys.executable, "-S", str(MAIN_PATH), option, value],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"argument {option}:" in result.stderr
    assert "Traceback" not in result.stderr


def test_help_does_not_require_gpu_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", str(MAIN_PATH), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--warmup-ms" in result.stdout
    assert "--rep-ms" in result.stdout
    assert "--restart" in result.stdout
    assert "results/run_state.json.tmp" in result.stdout
