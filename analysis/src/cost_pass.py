from __future__ import annotations

import re
import subprocess
from pathlib import Path


COST_FUNC_RE = re.compile(r"func\.func\s+@__cost_expr\b")


def pass_pipeline(func_name: str) -> str:
    return f"builtin.module(my-cost-analysis{{func-name={func_name}}})"


def run_cost_pass(
    triton_opt: Path,
    plugin: Path,
    ttgir_path: Path,
    func_name: str,
    timeout_s: float,
) -> str:
    command = [
        str(triton_opt),
        "--load-pass-plugin",
        str(plugin),
        "--pass-pipeline",
        pass_pipeline(func_name),
        str(ttgir_path),
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"triton-opt timed out after {timeout_s:g}s") from error
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            output.strip() or f"triton-opt failed with {result.returncode}"
        )
    return extract_cost_function(output)


def _brace_end(text: str, open_index: int) -> int | None:
    """Return the index of the brace at `open_index` that closes it, if any."""
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_cost_function(text: str) -> str:
    match = COST_FUNC_RE.search(text)
    if not match:
        raise ValueError("pass output did not contain func.func @__cost_expr")

    arrow = text.find("->", match.end())
    if arrow < 0:
        raise ValueError("cost function has no return type")

    # Skip the return-type parens (which may themselves contain per-result
    # `{cost.name = ...}` attribute dicts) to find the top-level brace that
    # starts either an optional `attributes { ... }` clause or the body.
    result_paren_depth = 0
    brace_start = None
    index = arrow + 2
    while index < len(text):
        char = text[index]
        if char == "(":
            result_paren_depth += 1
        elif char == ")":
            result_paren_depth -= 1
        elif char == "{" and result_paren_depth == 0:
            prefix = text[arrow + 2 : index].rstrip()
            if prefix.endswith("attributes"):
                end = _brace_end(text, index)
                if end is None:
                    break
                index = end + 1
                continue
            brace_start = index
            break
        index += 1
    if brace_start is None:
        raise ValueError("cost function has no body")

    body_end = _brace_end(text, brace_start)
    if body_end is None:
        raise ValueError("cost function body is unterminated")
    return text[match.start() : body_end + 1]
