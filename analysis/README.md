# Triton Cost Analysis

Generate static matplotlib plots for TTGIR cost estimates and benchmark results.

## Usage

```bash
uv run main.py
```

Plots are written to `plots/` by default. Use `--output-dir` to write them somewhere else:

```bash
uv run main.py --output-dir plots
```

The script reads benchmark data from `../data/result.json`, analyzes TTGIR files from `../data/ttgir`, and uses the pass plugin at `../build/libMyPass.so`.
