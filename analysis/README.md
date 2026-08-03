# Triton Cost Analysis

This directory parses the symbolic cost functions emitted by the C++ analysis
pass, converts them into hardware-pipeline work estimates, predicts kernel
runtime, and generates comparison plots.

## Cost function interface

The pass emits one `@__cost_expr` function with five named results:

```mlir
func.func @__cost_expr(...) -> (
  f64 {cost.name = "fp32"},
  f64 {cost.name = "fp64"},
  f64 {cost.name = "sfu"},
  f64 {cost.name = "tensor"},
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
- `memory`: loads, stores, and modeled local-memory operations

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
./run.sh 2>&1 | uv run --project analysis python analysis/main.py -
```

The Python API returns the same information as a dictionary:

```python
from parse import cost_equations

equations = cost_equations(cost_mlir)
memory_expression = equations["memory"]
```

`cost_equation()` remains available for legacy cost functions that return one
scalar result.

## Run the dataset analysis

From `analysis/`:

```bash
uv run analyze_costs.py \
  --results ../triton_gen/results/result.json \
  --ttgir-dir ../triton_gen/results/ttgir \
  --plugin ../build/libMyPass.so
```

Useful options include:

- `--limit N`: analyze only the first `N` benchmark records
- `--timeout SECONDS`: set the per-kernel `triton-opt` timeout
- `--gpu-spec PATH`: select GPU throughput and bandwidth data
- `--config PATH`: select operation weights and model settings
- `--output PATH`: select the prediction JSON destination
- `--scatter PATH`: select the measured-versus-predicted plot destination
- `--pipeline-plot PATH`: select the bottleneck-count plot destination

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
The prediction output records the scheduling model, inputs, block count, and
wave count. Future Python scheduling policies can therefore replace this model
without changing or rerunning the static operation analysis.

The predicted bottleneck is the category with the largest time. Total predicted
runtime is:

```text
launch overhead + max(fp32, fp64, sfu, tensor, memory category times)
```

Hardware rates and utilization factors come from `gpu_spec.json`.

## Outputs

The default JSON output is `cost_predictions.json`. Each successful prediction
contains:

- `category_expressions`: the five simplified symbolic equations
- `scheduled_work`: evaluated busiest-SM work for each category
- `schedule`: scheduler model, launch inputs, block count, and wave count
- `pipeline_ms`: estimated time for each category
- `bottleneck`: the category with the largest estimated time
- `predicted_ms` and `time_ms`: predicted and measured runtime

Generated figures are stored in `plots/` by default:

- `plots/cost_prediction_scatter.png`: measured runtime on the x-axis and
  predicted runtime on the y-axis, colored by predicted bottleneck
- `plots/cost_pipeline_counts.png`: number of kernels assigned to each
  bottleneck category

Both axes of the scatter plot use logarithmic scales. Its diagonal line marks
perfect agreement between measured and predicted runtime. Persistent matmul
kernels are excluded from plots until the cost analysis models their
runtime-dependent loop iterations.

## Tests

Run the focused parser, category, and plotting tests from `analysis/`:

```bash
uv run pytest -q tests/test_validation.py tests/test_category_costs.py tests/test_scheduler.py
```
