# Triton Cost Analysis

This directory parses the symbolic cost functions emitted by the C++ analysis
pass, converts them into hardware-pipeline work estimates, predicts kernel
runtime, and generates comparison plots.

## Cost function interface

The pass emits one `@__cost_expr` function with six named results:

```mlir
func.func @__cost_expr(...) -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
  f64 {cost.name = "l1"},
  f64 {cost.name = "memory"}
)
```

Each result is an independent symbolic equation:

- `fp32`: FP32 arithmetic, integer arithmetic, address generation, and modeled
  layout overhead
- `fp64`: operations producing FP64 results
- `sfu`: special functions, including MLIR math operations and external
  elementwise functions such as `__nv_asinf`
- `tensor`: tensor-core dot operations
- `l1`: modeled on-chip shared/L1 reads and writes
- `memory`: global-memory loads and stores

An asynchronous global-to-local copy contributes its payload bytes to both
`memory` and `l1`. Since category times are combined with `max`, this models the
same transfer at both levels without adding the payload latency twice.

The pass no longer collapses these categories into a scalar `Max` before
returning them. Control-flow alternatives can still produce `Max` expressions
inside an individual category.

## Build and inspect the pass

From the repository root:

```bash
cmake -S src -B build -G Ninja
cmake --build build
./run.sh
```

If `build/` was configured from a different checkout location, refresh it with:

```bash
cmake --fresh -S src -B build -G Ninja
cmake --build build
```

To simplify the emitted MLIR into one equation per category:

```bash
./run.sh 2>&1 | uv run --project analysis python -m analysis.src.parse -
```

The Python API returns the same information as a dictionary:

```python
from src.parse import cost_equations

equations = cost_equations(cost_mlir)
memory_expression = equations["memory"]
```

## Run the dataset analysis

From `analysis/`, pass the benchmark JSON and its TTGIR directory:

```bash
uv run main.py ../triton_gen/results/result.json ../triton_gen/results/ttgir
```

Use `--output DIR` to place the prediction JSON and both plots in one output
directory. It defaults to `analysis/output/`.

To analyze the checked-in local data with the default output directory, run:

```bash
./run_analysis.sh
```

The analyzer uses `gpu_spec.json`, `cost_analysis_config.json`, and
`../build/libMyPass.so`, and finds `triton-opt` on `PATH`.

## Prediction model

The C++ pass remains independent of the host launch grid. It converts each
per-thread category equation to per-block work using TTGIR metadata:

```text
threads per block = num warps * threads per warp
per-block work = per-thread work * threads per block
```

The Python `schedule_work()` function in `scheduler.py` then combines that
static work with the recorded host launch and the selected GPU specification:

```text
program count = product(grid size)
total blocks = program count * CTAs per program
waves = ceil(total blocks / GPU SM count)
scheduled SM work = per-block work * waves
```

This scheduled work is the critical-path work assigned to the busiest SM. The
model assumes each SM executes one block at a time. It intentionally does not
model additional resident blocks, latency hiding, register pressure, or shared
memory occupancy yet.

The analyzer converts scheduled work to time using per-SM category capacities:

```text
category time = scheduled SM work / per-SM category capacity
```

Global memory bandwidth is currently divided evenly across the GPU's SMs. This
is an explicit simplifying assumption, not a measured per-SM hardware property.
The L1 rate uses the configured per-SM bytes per cycle and clock. For the current
Ampere specification, its 128 bytes/cycle represents the aggregate 32-bank,
32-bit shared-memory data path exposed through the unified L1/shared-memory
hardware. It does not model cache-hit rates, bank conflicts, or instruction
issue limits yet. The prediction output records the scheduling model, inputs,
block count, and wave count. Future Python scheduling policies can therefore
replace this model without changing or rerunning the static operation analysis.

The predicted bottleneck is the category with the largest time. Total predicted
runtime is:

```text
launch overhead + max(fp32, fp64, sfu, tensor, l1, memory category times)
```

Hardware rates and utilization factors come from `gpu_spec.json`. The Python
implementation is split by responsibility:

- `main.py`: CLI and output coordination
- `src/analyzer.py`: dataset orchestration
- `src/cost_pass.py`: `triton-opt` execution and emitted-function extraction
- `src/cost_model.py`: pipeline evaluation and throughput calculations
- `src/results.py`: result types, summaries, and JSON output
- `src/plotting.py`: prediction plots

Root `main.py` keeps the command-line interface separate from the reusable
implementation under `src/`.

## Outputs

The default JSON output is `output/cost_predictions.json`. Each successful
prediction contains:

- `category_expressions`: the six simplified symbolic equations
- `scheduled_work`: evaluated busiest-SM work for each category
- `schedule`: scheduler model, launch inputs, block count, and wave count
- `pipeline_ms`: estimated time for each category
- `bottleneck`: the category with the largest estimated time
- `predicted_ms` and `time_ms`: predicted and measured runtime

Generated figures are stored in `output/` by default:

- `output/cost_prediction_scatter.png`: measured runtime on the x-axis and
  predicted runtime on the y-axis, colored by kernel with marker shapes for
  predicted bottlenecks
- `output/cost_pipeline_counts.png`: number of configurations assigned to each
  bottleneck category, stacked and colored by kernel

Both axes of the scatter plot use logarithmic scales. Its solid diagonal marks
perfect agreement, and the dashed diagonals show the 2x error boundary.
Persistent matmul kernels are excluded from plots until the cost analysis
models their runtime-dependent loop iterations.

## Tests

Run the test suite from `analysis/`:

```bash
uv run pytest -q
```
