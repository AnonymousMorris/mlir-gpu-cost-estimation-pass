from __future__ import annotations

import operator
from pathlib import Path
import sys
import tempfile

import sympy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation"))

from src.cost_pass import run_cost_pass
from src.parse import parse_cost_analysis
from cost_mlir import parse_cost_function as validation_parse
from equation import build_equations as validation_equations


def main() -> None:
    triton_opt, plugin, fixture = map(Path, sys.argv[1:])
    load = sympy.Symbol("triton.load_cost")
    store = sympy.Symbol("triton.store_cost")
    add = sympy.Symbol("arith.addf32")

    def analyze(path: Path, name: str):
        mlir = run_cost_pass(triton_opt, plugin, path, name, timeout_s=10.0)
        parsed = parse_cost_analysis(mlir)
        independent = validation_equations(validation_parse(mlir))
        for category, expression in parsed.equations.items():
            assert sympy.simplify(expression - independent[category]) == 0
        return parsed

    for name, bytes_per_direction in (
        ("constant_bound", 37 * 2),
        # Preserve the existing lane/replication accounting: 32 lanes * 8 bytes
        # times (all true + all false + 3/8 true), not unique-address counting.
        ("constant_masks", 32 * 8 * (1 + 3 / 8)),
        ("scalar_masks", 32 * 4),
    ):
        parsed = analyze(fixture, name)
        assert parsed.memory_mask_model == "exact"
        assert not parsed.memory_mask_approximations
        assert not parsed.runtime_symbols
        assert sympy.simplify(
            parsed.equations["memory"] - bytes_per_direction * (load + store)
        ) == 0

    for name in ("range_mask", "independent_loop", "broadcast_mask"):
        parsed = analyze(fixture, name)
        assert parsed.memory_mask_model == "exact"
        assert parsed.runtime_symbols == {"N"}
        for n in (-5, 0, 1, 37, 127, 128, 129):
            # load + store: f32 * one row; f32 * three iterations; f16 * four rows.
            bytes_per_element = {"range_mask": 8, "independent_loop": 24,
                                 "broadcast_mask": 16}[name]
            actual = parsed.equations["memory"].subs({load: 1, store: 1, "N": n})
            assert float(actual) == bytes_per_element * max(0, min(n, 128))
        if name != "broadcast_mask":
            # A false mask does not erase arithmetic on the loaded tensor.
            expected_adds = 128 * (3 if name == "independent_loop" else 1)
            assert float(parsed.equations["fp32"].expand().coeff(add)) == expected_adds

    for name in ("dependent_loop", "program_mask", "unknown_mask",
                 "nonuniform_bound", "branch_bound", "async_mask"):
        parsed = analyze(fixture, name)
        assert parsed.memory_mask_model == "upper_bound"
        assert parsed.memory_mask_approximations
        assert not parsed.runtime_symbols  # no fabricated IV/pid/unknown bindings
        if name == "async_mask":
            assert sympy.simplify(parsed.equations["memory"] - 512 * load) == 0
            assert sympy.simplify(parsed.equations["l1"] - 512 * load) == 0  # zero fill unchanged
            assert "async copy" in parsed.memory_mask_approximations[0]
        else:
            repetitions = 3 if name == "dependent_loop" else 1
            assert sympy.simplify(
                parsed.equations["memory"] - 512 * repetitions * (load + store)
            ) == 0
            assert len(parsed.memory_mask_approximations) == 2  # deduplicated by op/reason

    # Exercise strict/inclusive endpoints, swapped operands, nonzero starts,
    # signed extremes, and unsigned interpretation of a negative i32 bit pattern.
    comparisons = {
        "slt": operator.lt, "sle": operator.le,
        "sgt": operator.gt, "sge": operator.ge,
        "ult": operator.lt, "ule": operator.le,
        "ugt": operator.gt, "uge": operator.ge,
        "eq": operator.eq, "ne": operator.ne,
    }
    source = fixture.read_text()
    with tempfile.TemporaryDirectory() as directory:
        variant = Path(directory) / "mask.ttgir"
        for predicate, compare in comparisons.items():
            for swapped in (False, True):
                text = source.replace("arith.cmpi slt, %range, %limit :",
                                      f"arith.cmpi {predicate}, " +
                                      ("%limit, %range" if swapped else "%range, %limit") + " :")
                text = text.replace("end = 128 : i32, start = 0 : i32",
                                    "end = 160 : i32, start = 32 : i32")
                variant.write_text(text)
                parsed = analyze(variant, "range_mask")
                assert parsed.memory_mask_model == "exact"
                for n in (-(2**31), -1, 0, 32, 33, 95, 159, 160, 2**31 - 1):
                    bound = n % 2**32 if predicate.startswith("u") else n
                    active = sum(compare(bound, i) if swapped else compare(i, bound)
                                 for i in range(32, 160))
                    actual = parsed.equations["memory"].subs({load: 1, store: 1, "N": n})
                    assert float(actual) == active * 8, (predicate, swapped, n, actual)

        # Unsigned comparisons of ranges crossing zero are intentionally outside
        # this small recognizer, not treated as a signed interval accidentally.
        variant.write_text(source.replace("arith.cmpi slt", "arith.cmpi ult").replace(
            "end = 128 : i32, start = 0 : i32", "end = 96 : i32, start = -32 : i32"
        ))
        assert analyze(variant, "range_mask").memory_mask_model == "upper_bound"


if __name__ == "__main__":
    main()
