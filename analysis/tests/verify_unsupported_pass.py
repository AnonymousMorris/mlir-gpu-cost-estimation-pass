from __future__ import annotations

import resource
import sys
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.cost_model import pipeline_exprs
from src.cost_pass import run_cost_pass
from src.parse import cost_equations


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_unsupported_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )
    # Unknown operations intentionally terminate the pass. Do not leave cores.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    for function in (
        "unsupported_scalar", "unsupported_tensor", "unsupported_reduction_combiner",
    ):
        try:
            run_cost_pass(triton_opt, plugin, ttgir, function, timeout_s=10.0)
        except RuntimeError as error:
            message = str(error)
            assert "unknown op in cost analysis: arith.negf" in message, message
            assert "func.func @__cost_expr" not in message, message
        else:
            raise AssertionError(f"{function} silently accepted an unsupported op")

    cost_mlir = run_cost_pass(
        triton_opt, plugin, ttgir, "explicit_zero", timeout_s=10.0
    )
    equations = pipeline_exprs(cost_equations(cost_mlir))
    assert all(value.is_zero for value in equations.values()), equations


if __name__ == "__main__":
    main()
