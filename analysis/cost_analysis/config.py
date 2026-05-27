from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULT_JSON = DATA_ROOT / "result.json"
TTGIR_ROOT = DATA_ROOT / "ttgir"
PASS_PLUGIN = PROJECT_ROOT / "build" / "libMyPass.so"
