from __future__ import annotations

from pathlib import Path
import sys


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.cost_pass import run_cost_pass
from src.parse import cost_equations


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_l1_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )

    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    cost_mlir = run_cost_pass(
        triton_opt,
        plugin,
        ttgir,
        "tiled_matmul_kernel",
        timeout_s=10.0,
    )
    equations = cost_equations(cost_mlir)

    l1_symbols = {str(symbol) for symbol in equations["l1"].free_symbols}
    memory_symbols = {
        str(symbol) for symbol in equations["memory"].free_symbols
    }
    local_symbols = {
        "triton_gpu.local_alloc_cost",
        "triton_gpu.local_load_cost",
    }

    if not local_symbols <= l1_symbols:
        missing = ", ".join(sorted(local_symbols - l1_symbols))
        raise SystemExit(f"l1 expression is missing local costs: {missing}")
    leaked = local_symbols & memory_symbols
    if leaked:
        names = ", ".join(sorted(leaked))
        raise SystemExit(f"global-memory expression contains local costs: {names}")
    if "triton.load_cost" not in memory_symbols:
        raise SystemExit("global-memory expression is missing load bytes")

    has_async_copy = "ttg.async_copy_global_to_local" in ttgir.read_text()
    if has_async_copy and "triton.load_cost" not in l1_symbols:
        raise SystemExit("l1 expression is missing async-copy destination bytes")


if __name__ == "__main__":
    main()
