# Triton Cost Analysis

This directory parses the symbolic cost functions emitted by the C++ analysis
pass, converts them into hardware-pipeline work estimates, predicts kernel
runtime, and generates comparison plots.

## Cost function interface

The pass emits one `@__cost_expr` function with seven named results and explicit
work units:

```mlir
func.func @__cost_expr(...) -> (
  f64 {cost.name = "fp32", cost.unit = "instructions"},
  f64 {cost.name = "fp64", cost.unit = "instructions"},
  f64 {cost.name = "int32", cost.unit = "instructions"},
  f64 {cost.name = "sfu", cost.unit = "instructions"},
  f64 {cost.name = "tensor", cost.unit = "flops"},
  f64 {cost.name = "l1", cost.unit = "bytes"},
  f64 {cost.name = "memory", cost.unit = "bytes"}
)
```

Symbolic function arguments are tagged by role. Operation-cost weights use
`cost.kind = "weight"`, while kernel values used by symbolic control flow use
`cost.kind = "runtime"`. The analyzer substitutes runtime symbols from the
benchmark record and substitutes each required weight from
`parameters.<symbol>.work_per_count`. Missing runtime bindings, missing weights,
and negative/nonfinite weights are rejected; there is no implicit unit-1 fallback.

Each result is an independent symbolic equation:

| Category | Work unit | Interpretation |
|---|---|---|
| `fp32` | Per-lane instruction-equivalents | Scalar FP arithmetic and current movement/conversion proxies |
| `fp64` | Per-lane instruction-equivalents | Scalar FP64 arithmetic |
| `int32` | Per-lane instruction-equivalents | Integer/index arithmetic, comparisons, ranges, and pointer arithmetic |
| `sfu` | Per-lane instruction-equivalents | Special functions and external elementwise functions |
| `tensor` | FLOPs | Two FLOPs per dot multiply-accumulate |
| `l1` | Bytes | Logical shared-memory payload |
| `memory` | Bytes | Logical global-memory payload |

Scalar add, multiply, and FMA each count as **one instruction-equivalent**.
For example, 128 scalar FMAs require one ideal cycle at 128 instructions/SM/cycle,
while 512 tensor multiply-accumulates require one ideal cycle at 1024 FLOPs/SM/cycle.
The registered MLIR FMA operation is `math.fma`; its weight retains the historical
name `arith.fma32` (or `16`/`64`).

Weights are explicit work multipliers in the result category's unit. Operations
whose expansion or communication cost has not been modeled must not silently
receive weight 1. The checked-in configuration marks division/remainder
(including ceiling division), external `__nv_asinf`, and generic layout
conversion with `work_per_count: null` and an `unmodeled_reason`. If their symbols
occur in the evaluated equations, analysis records a skip with that reason.

To opt into a calibrated approximation, replace the null with an explicit finite,
nonnegative `work_per_count` in `cost_analysis_config.json`. This is a deliberate
model choice, not automatic fallback. Layout-conversion proxies still consume
the FP32 instruction budget; supplying a weight does not infer their physical
communication path. FP16 packing, 64-bit integer operations, compiler fusion,
and reduction shuffles remain coarse approximations.

Unsupported IR operations are different from uncalibrated symbolic costs:
the C++ pass rejects them for scalar and tensor operands alike, including inside
nested regions. Explicit zero-cost entries (such as constants and shape-only
broadcasts) remain supported; an absent handler never means zero cost.

Global-memory payloads are scaled by their element types: for example, FP16
values contribute two bytes per element and FP32 values contribute four. This
is logical payload accounting; masks, cache lines, alignment, coalescing, and
partially used memory transactions are not modeled yet.

An asynchronous global-to-local copy contributes its type-scaled payload bytes
to both `memory` and `l1`. Since category times are combined with `max`, this
models the same transfer at both levels without adding the payload latency
twice.

A `tt.reduce` derives thread-local values, participating warp lanes, and
remaining inter-warp values from the operand's linear layout. The current model
counts thread-local combines and the target-independent warp shuffle tree. Warp
communication is represented by the `triton.reduce_shuffle_cost` weight.
Target-specific packing/redux instructions and inter-warp layout conversions
are not modeled yet; the inter-warp stage is an explicit zero-cost stub.

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

Benchmark records containing symbolic runtime-dependent loops must include the
named `scalar_args` metadata produced by the current `triton_gen/main.py`.
Regenerate older benchmark data that predates this field.

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

### Rates and the GA10x shared execution constraint

Hardware capacities are separated by unit:

- `per_sm_instructions_per_cycle`: FP32, FP64, INT32, SFU, and the optional
  combined `fp32_int32` execution capacity.
- `per_sm_flops_per_cycle`: tensor paths such as `tensor_tf32` and `tensor_fp16`.
- `per_sm_bytes_per_cycle`: shared-memory bandwidth.

All capacities are multiplied by clock and the corresponding utilization factor.
FP64 capacity is explicit, not inferred from FP32. `tensor_throughput_key` still
selects one tensor path for the whole analysis; per-dot precision selection is
not modeled.

For GA10x, FP32-only lanes coexist with FP32/INT32-capable lanes. In addition to
the individual resource times, the analyzer calculates:

```text
fp32_int32 time = (fp32 instructions + int32 instructions) / combined capacity
```

The RTX 3060 specification uses 128 FP32, 64 INT32, and 128 combined per-lane
instructions/SM/cycle before utilization. Thus 128 FP32 instructions plus 64
integer instructions require at least 1.5 ideal cycles, not one. This is an
aggregate capacity bound, not a scheduler or latency simulator. The combined
resource has no additional work equation and is not added to either work count.
Omit its capacity for architectures without this sharing arrangement (do not
apply the GA10x values to GA100).

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
launch overhead + max(fp32, fp64, int32, sfu, tensor, l1, memory,
                      optional fp32_int32 resource time)
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

- `category_expressions`: the seven simplified symbolic work equations
- `scalar_args`: named numeric launch arguments used for symbolic evaluation
- `scheduled_work`: evaluated busiest-SM work for each category
- `schedule`: scheduler model, launch inputs, block count, and wave count
- `pipeline_ms`: estimated time for each category and the optional `fp32_int32`
  shared-resource constraint
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

## Configuration migration

Rebuild the pass and regenerate predictions after changing to this contract.
Old six-result cost functions are rejected by the analyzer.

- Replace `ops_per_count` with `work_per_count`; scalar FMA weights are now 1,
  not 2. Dot equations themselves now contain the factor of 2 for FLOPs.
- Replace `per_sm_ops_per_cycle` with the unit-specific capacity dictionaries.
- Removed ignored fields: parameter `pipeline`, `defaults`, `operation_paths`,
  `memory_bytes_per_count`, and `l1_bytes_per_count`.
- Resource assignment remains in C++; JSON weights only scale work.
- Unmodeled division/remainder, external asin, and layout-conversion weights are
  null until explicitly calibrated. Runs may therefore report more skips than
  older versions that assigned them placeholder weight 1.
- Prediction JSON no longer includes the legacy `expression = Max(work...)`
  field, which compared incompatible units. Use `category_expressions` instead.

## Tests

Run the test suite from `analysis/`:

```bash
uv run pytest -q
```
