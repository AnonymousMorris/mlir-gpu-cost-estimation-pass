from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import sympy

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.cost_model import eval_work, pipeline_exprs
from src.cost_pass import run_cost_pass
from src.parse import parse_cost_analysis


# Per-thread golden counts: shuffles, data selects/permutes, shared bytes,
# warp synchronizations, CTA barriers. Block size is 128 threads.
Counts = tuple[int, int, int, int, int]
COUNTS: dict[str, Counts] = {
    "identity": (0, 0, 0, 0, 0),
    "equivalent_encodings": (0, 0, 0, 0, 0),
    "register_only": (0, 0, 0, 0, 0),
    "register_broadcast": (0, 0, 0, 0, 0),
    "warp_register_broadcast": (1, 3, 0, 0, 0),
    "lane_permutation": (1, 0, 0, 0, 0),
    "warp_ship": (16, 48, 0, 0, 0),
    "warp_swap": (32, 64, 0, 0, 0),
    "mma_to_blocked": (0, 0, 512, 0, 15),
    "shared_cta": (0, 0, 64, 0, 1),
    "shared_warp_multi": (0, 0, 256, 15, 0),
    "forced_warp_multi": (24, 128, 0, 0, 0),
    "shared_register_broadcast": (0, 0, 32, 0, 1),
}


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_layout_conversion_pass.py TRITON_OPT PASS_PLUGIN TTGIR"
        )
    triton_opt, plugin, ttgir = map(Path, sys.argv[1:])
    config = json.loads((ANALYSIS_ROOT / "cost_analysis_config.json").read_text())
    kinds = ("shuffle", "select", "shared_byte", "warp_sync", "cta_barrier")
    symbols = [sympy.Symbol(f"triton_gpu.layout.{kind}_cost") for kind in kinds]

    def check(fixture: Path, function: str, counts: Counts) -> None:
        parsed = parse_cost_analysis(run_cost_pass(
            triton_opt, plugin, fixture, function, timeout_s=10.0
        ))
        equations = pipeline_exprs(parsed.equations)
        shuffles, selects, shared_bytes, warp_syncs, barriers = counts
        expected = {
            "int32": 128 * (shuffles * symbols[0] + selects * symbols[1] +
                            warp_syncs * symbols[3] + barriers * symbols[4]),
            "l1": 128 * shared_bytes * symbols[2],
        }
        for category, expression in equations.items():
            assert sympy.simplify(expression - expected.get(category, 0)).is_zero, (
                function, category, expression
            )
        work = eval_work(
            equations, config, {}, weight_symbols=parsed.weight_symbols,
            runtime_symbols=parsed.runtime_symbols,
        )
        assert work["int32"] == 128 * (shuffles + selects + warp_syncs + barriers)
        assert work["l1"] == 128 * shared_bytes
        assert work["fp32"] == work["memory"] == 0

    def lower(fixture: Path) -> dict[str, str]:
        result = subprocess.run(
            [str(triton_opt), str(fixture),
             "--allocate-shared-memory-nv=compute-capability=86 ptx-version=83",
             "--convert-triton-gpu-to-llvm=compute-capability=86 ptx-version=83",
             "--convert-nv-gpu-to-llvm"],
            text=True, capture_output=True, check=True, timeout=10.0,
        )
        return dict(re.findall(
            r'llvm.func[^@]*@(\w+)(.*?)(?=\n  llvm.func|\Z)', result.stdout, re.S
        ))

    def check_lowering(body: str, counts: Counts, check_selects: bool = True) -> None:
        shuffles, selects, shared_bytes, warp_syncs, barriers = counts
        assert body.count("nvvm.shfl.sync") == shuffles
        if check_selects and not shared_bytes:
            assert body.count("llvm.select") == selects
        assert body.count("nvvm.bar.warp.sync") == warp_syncs
        assert body.count("nvvm.barrier0") == barriers
        payload = 0
        for line in body.splitlines():
            if "llvm.store " in line or "llvm.load " in line:
                vector = re.search(r'vector<(\d+)x[if](\d+)>', line)
                if vector:
                    payload += int(vector[1]) * int(vector[2]) // 8
                else:
                    scalar = re.search(r'(?:-> |: )[if](\d+)(?:,|$)', line)
                    assert scalar, line
                    payload += int(scalar[1]) // 8
            elif "nvvm.ldmatrix " in line:
                matrices = re.search(r'num = (\d+) : i32', line)
                assert matrices, line
                payload += 4 * int(matrices[1])  # i32 results per lane.
        assert payload == shared_bytes, (payload, shared_bytes)

    for function, counts in COUNTS.items():
        check(ttgir, function, counts)
    native = lower(ttgir)
    assert set(native) == set(COUNTS)
    for function, counts in COUNTS.items():
        check_lowering(native[function], counts)

    source = ttgir.read_text()
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "case.ttgir"
        # Includes sub-byte widening and pointer width independent of pointee.
        for element, ship, shared in (
            ("f16", (8, 24, 0, 0, 0), (0, 0, 32, 0, 1)),
            ("i8", (4, 12, 0, 0, 0), (0, 0, 16, 0, 1)),
            ("i1", (4, 12, 0, 0, 0), (0, 0, 16, 0, 1)),
            ("f64", (32, 96, 0, 0, 0), (0, 0, 128, 0, 1)),
            ("!tt.ptr<i8>", (32, 96, 0, 0, 0), (0, 0, 128, 0, 1)),
        ):
            fixture.write_text(source.replace("xf32", "x" + element))
            check(fixture, "warp_ship", ship)
            check(fixture, "shared_cta", shared)
            lowered = lower(fixture)
            # LLVM selects for packed/wide types need not match 32-bit word
            # instruction counts. Shuffle counts and actual bytes must match.
            check_lowering(lowered["warp_ship"], ship, check_selects=False)
            check_lowering(lowered["shared_cta"], shared)

        for target in ("80", "87", "90"):
            fixture.write_text(source.replace("cuda:86", f"cuda:{target}"))
            check(fixture, "register_only", COUNTS["register_only"])
            if target != "90":
                check(fixture, "shared_cta", COUNTS["shared_cta"])
                check(fixture, "warp_ship", COUNTS["warp_ship"])
            else:
                expect_failure(fixture, "shared_cta", "require an Ampere target")

        generic = source.replace("#ttg.linear<", "#ttg.generic_linear<")
        # tt.return does not accept GenericLinearEncoding operands. These
        # cost-only negative cases need no live results or function returns.
        lines = []
        for line in generic.splitlines():
            if line.startswith("  tt.func"):
                line = re.sub(r' -> tensor<.*>(?= (?:attributes|\{))', "", line)
            elif line.startswith("    tt.return %"):
                line = "    tt.return"
            lines.append(line)
        fixture.write_text("\n".join(lines) + "\n")
        check(fixture, "register_only", COUNTS["register_only"])
        expect_failure(fixture, "shared_register_broadcast", "generic-linear")
        expect_failure(fixture, "lane_permutation", "generic-linear")

        fixture.write_text(source.replace("xf32", "xi4"))
        expect_failure(fixture, "warp_ship", "unmodeled element bit width")

        fixture.write_text(source.replace(
            "-> tensor<32x32xf32, #shared_dst> {",
            "-> tensor<32x32xf32, #shared_dst> attributes {always_use_warp_shuffle} {",
        ))
        expect_failure(fixture, "shared_cta", "forced warp shuffle")

        fixture.write_text('''#src = #ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32]], warp = [[64], [128]], block = [[256]]}>
#dst = #ttg.linear<{register = [[256]], lane = [[2], [4], [8], [16], [32]], warp = [[64], [128]], block = [[1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @cross_cta(%x: tensor<512xf32, #src>) {
    %y = ttg.convert_layout %x : tensor<512xf32, #src> -> tensor<512xf32, #dst>
    tt.return
  }
}
''')
        expect_failure(fixture, "cross_cta", "cross-CTA")


def expect_failure(fixture: Path, function: str, reason: str) -> None:
    triton_opt, plugin = map(Path, sys.argv[1:3])
    try:
        run_cost_pass(triton_opt, plugin, fixture, function, timeout_s=10.0)
    except RuntimeError as error:
        assert "unsupported layout conversion" in str(error), str(error)
        assert reason in str(error), str(error)
    else:
        raise AssertionError(f"silently modeled unsupported {function}")


if __name__ == "__main__":
    main()
