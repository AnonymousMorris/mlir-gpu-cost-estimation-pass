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
| `int32` | Per-lane instruction-equivalents | Integer/index arithmetic, comparisons, ranges, pointer arithmetic, and layout movement proxies |
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
receive weight 1. The checked-in configuration marks unmodeled integer
widths/targets, unsigned and ceiling division/remainder, unmodeled floating-division
variants, and external `__nv_asinf` with
`work_per_count: null` and an
`unmodeled_reason`. If their symbols occur in the evaluated equations, analysis
records a skip with that reason.

To opt into a calibrated approximation, replace the null with an explicit finite,
nonnegative `work_per_count` in `cost_analysis_config.json`. This is a deliberate
model choice, not automatic fallback. Layout conversion instead has the explicit
path-sensitive model below. General FP16 arithmetic packing, 64-bit integer
operations, compiler fusion, and reduction shuffles remain coarse approximations.

Unsupported IR operations are different from uncalibrated symbolic costs:
the C++ pass rejects them for scalar and tensor operands alike, including inside
nested regions. Explicit zero-cost entries (such as constants and shape-only
broadcasts) remain supported; an absent handler never means zero cost.

### Structural views and floating-point extension

- `tt.bitcast` is zero-cost register reinterpretation. Triton's verifier enforces
  matching scalar bit widths, or pointer-to-pointer casts in the same address
  space, as well as tensor shape/encoding compatibility.
- `tt.trans` is zero-cost at the TTGIR boundary: it permutes shape and encoding
  without changing register ownership. Surrounding `ttg.convert_layout` operations
  retain their separate path-sensitive movement cost.
- `arith.extf` has source/destination-specific weights, not a generic unit cost.
  `arith.extf.f16_to_f32` counts one floating conversion per value in the FP32
  instruction budget. `arith.extf.bf16_to_f32` counts one integer shift per value
  in the INT32 budget. Both scale by elements per thread and threads per block.
  The FP32 budget is a coarse conversion-throughput proxy, not a measured
  dedicated conversion-pipeline rate.
- Extensions to FP64 have explicit null weights until calibrated. Other
  unconfigured type pairs fail weight lookup rather than receiving cost 1.
  Conversion to FP64 is not automatically charged at FP64 arithmetic throughput.

These rules follow Triton's `BitcastOp::verify` in
`lib/Dialect/Triton/IR/Ops.cpp`, `TransOpConversion` in
`lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp`, and the elementwise
`LLVM::BitcastOp` / `LLVM::FPExtOp` conversions. SM86 lowering of the live scalar
extension fixtures produces `cvt.f32.f16` and `cvt.f32.bf16` in PTX; the checked
CUDA assembler lowers them to a floating conversion (`HADD2.F32`) and an
integer shift (`SHF.L.U32`), respectively. Packing, deduplication, and fusion
with neighboring operations are not modeled.

### Ampere layout conversion

`src/LayoutConversion.cpp` models `ttg.convert_layout` separately from tensor
arithmetic. It uses Triton's `minimalCvtLayout` and warp decomposition helpers,
not a universal per-element conversion weight:

- Identical register ownership and within-thread register reordering are zero
  work (SSA rewiring), including equivalent encodings and register broadcasts.
- Warp-only movement follows the backend's shuffle-versus-shared choice and
  Ship/Swap algorithms. Counts include packing into 32-bit words, data
  selection/permutation, and splitting 64-bit values into two shuffle words.
  An enclosing `always_use_warp_shuffle` attribute follows the backend's policy.
- Shared-memory paths count source-store plus destination-load payload bytes,
  after removing register broadcasts. `i1` is widened to `i8`. Scratch repetition
  selection includes Ampere's `ldmatrix` load tiles, not Hopper's `stmatrix`.
  Each scratch repetition needs a synchronization after stores; scratch reuse
  adds another, giving `2 * repetitions - 1` warp syncs or CTA barriers.
- Inter-thread movement supports `cuda:80`, `cuda:86`, and `cuda:87` with
  1/8/16/32/64-bit elements, including 64-bit pointers. Cross-CTA movement,
  generic-linear inter-thread conversions, other widths/targets, and forced
  warp shuffles that cross warps fail explicitly, without a generic fallback.

| Parameter suffix (under `triton_gpu.layout.`) | Category/unit |
|---|---|
| `shuffle_cost` | INT32 instruction-equivalents per 32-bit shuffle |
| `select_cost` | INT32 instruction-equivalents per word select/permute |
| `shared_byte_cost` | L1 bytes per payload byte |
| `warp_sync_cost` | INT32 instruction-equivalents per issued warp sync |
| `cta_barrier_cost` | INT32 instruction-equivalents per issued CTA barrier |

These explicit per-unit multipliers default to 1; missing weights still fail.
Work is per-thread and receives the existing block scaling exactly once.
Instruction counts use the INT32 capacity as a **coarse throughput proxy**, not
measured shuffle/barrier throughput. The model does not simulate barrier wait
latency, shared-memory bank conflicts, scratch address arithmetic, packing
instructions outside the movement algorithm, occupancy, or compiler fusion.

The policy follows `lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp`,
NVIDIA's `ConvertLayoutOpSwizzlingConversion`, and `getSrcDstTiles` in Triton's
conversion utilities. `verify_layout_conversion_pass.py` checks symbolic counts
and live-result NVIDIA LLVM lowering, including ordinary shared loads/stores,
`ldmatrix`, scratch reuse, register broadcasts, packing, and pointer width.

### Floating-point comparisons

`arith.cmpf` is classified by the compared operands, not its `i1` result.
FP32 operands contribute `arith.cmpf32` work to the FP32 instruction budget;
they never acquire an integer-comparison cost just because the result is boolean.
Tensor comparisons scale by elements per thread and threads per block.

The SM86 model counts one `FSETP` instruction-equivalent per FP32 comparison for
all 14 nonconstant predicates: `oeq`, `one`, `ogt`, `oge`, `olt`, `ole`, `ord`,
`ueq`, `ugt`, `uge`, `ult`, `ule`, `une`, and `uno`. Ordered/unordered NaN
semantics are preserved by Triton's `CmpFOpConversion` to `LLVM::FCmpOp`.
Checked SM86 PTX/SASS lowering uses a single floating comparison for each,
sometimes with an inverted predicate at its consumer. This uses the existing
aggregate FP32 capacity, not a separately calibrated predicate pipeline.

The `true` and `false` predicates contribute zero comparison work, including
for otherwise uncalibrated operand types. This does not erase work for nearby
selects or stores. Self-comparisons such as `oeq x, x` are not assumed constant:
NaN inputs still matter. General constant folding and compare/consumer fusion
are not modeled.

FP64 comparisons are correctly classified as FP64 work, with an explicit null
`arith.cmpf64` weight until calibrated. FP16/BF16 comparisons likewise retain a
null `arith.cmpf16` weight. Neither falls back to the FP32 unit weight, and
`arith.cmpi` continues to use the INT32 budget.

### Ampere FP32 division

For `cuda:80`, `cuda:86`, and `cuda:87`, FP32 `arith.divf` contributes an
instruction mix to both FP32 and SFU work. It does not use a single generic
division weight:

| Case | FP32 instructions/value | SFU instructions/value |
|---|---:|---:|
| General `x / y` | 7 | 1 |
| Statically known `1 / y` | 6 | 1 |

The four weights are `arith.divf32.fp32_cost`, `arith.divf32.sfu_cost`,
`arith.reciprocal32.fp32_cost`, and `arith.reciprocal32.sfu_cost`.
Scalar constants, dense splat constants, and `tt.splat` of scalar one identify
the reciprocal case. A dynamic numerator, even if its runtime value happens to
be one, uses the general model. Existing work for the `tt.splat` itself is
unchanged. All components scale by elements per thread and threads per block.

Triton's `FDivOpConversion` lowers both cases to `llvm.nvvm.div.full` /
`div.full.f32`, not just an unchecked reciprocal instruction. The checked
SM80/SM86/SM87 SASS expansions contain:

- General division: 2 `FSETP`, 3 unconditional `FMUL`, 2 `FSEL`, 1 `MUFU.RCP`.
- Reciprocal: 2 `FSETP`, 2 unconditional `FMUL`, 2 `FSEL`, 1 `MUFU.RCP`.

This is a **normal-path throughput approximation**, not a latency or exact
instruction-issue model. FP32 work groups the floating comparisons/selects with
arithmetic. Two additional predicated `FMUL` instructions for subnormal-range
rescaling are not counted. Register/constant materialization, compiler
constant-denominator strength reduction, fusion, and common-subexpression
elimination are also not modeled. Do not treat these counts as an upper bound
for exceptional inputs or as a measured division latency.

FP64 division (`arith.divf64`), FP16/BF16 division (`arith.divf16`), and FP32
division outside the supported Ampere targets (`arith.divf32`) retain explicit
null weights. Unsupported variants cannot inherit the FP32/SFU mix implicitly.
The old untyped `arith.divf` configuration parameter has been replaced.

### Ampere signed i32 division and remainder

`arith.divsi` and `arith.remsi` have a dedicated per-lane model for i32 on
`cuda:80`, `cuda:86`, and `cuda:87`. Tensor/block scaling remains unchanged.
The pass distinguishes compile-time scalar/dense-splat divisors (including
`tt.splat` of a scalar constant) from runtime or nonuniform tensor divisors.
Recorded runtime argument values are not mistaken for compiler constants.

Constant divisors contribute only INT32 work:

| Divisor | Division instructions/value | Remainder instructions/value |
|---|---:|---:|
| `1` | 0 | 0 |
| `-1` | 1 (negation, valid inputs) | 0 |
| `2` / `-2` | 2 / 3 | 3 |
| `2^k` / `-2^k`, `k > 1` | 3 / 4 | 4 |
| `INT_MIN` | 2 (compare/select) | 4 |
| Other constants | Magic multiply-high, shift, rounding correction | Same quotient sequence plus fused multiply/subtract |

Signed powers of two include bias for truncation toward zero; they are not
modeled as a single arithmetic shift. General constants use LLVM's
`SignedDivisionByConstantInfo`, not a hand-written magic-number algorithm.
The estimated fused Ampere quotient count is `2 + (shift != 0)`, plus one for
negative-divisor/positive-magic numerator correction. Remainder uses the positive
magnitude's quotient sequence plus one `IMAD` for `x - q*d`, independent of the
divisor's sign. The `*.constant.int32_cost` weights are per-instruction multipliers
(default 1), not one instruction per division. Fully constant scalar/splat inputs
fold to zero runtime arithmetic work after checking invalid cases.

Runtime divisors use this conservative valid-input instruction mix:

| Operation | INT32 work/value | FP32 work/value | SFU work/value |
|---|---:|---:|---:|
| Signed i32 division | 18 | 2 | 1 |
| Signed i32 remainder | 16 | 2 | 1 |

The weights are `arith.divsi32.runtime.{int32,fp32,sfu}_cost` and the corresponding
`arith.remsi32` names. Checked SM80/SM86/SM87 lowering goes through LLVM `sdiv` /
`srem` to PTX `div.s32` / `rem.s32`; SASS includes `IABS`, integer refinement,
comparisons/corrections, `I2F`, `MUFU.RCP`, and `F2I`. Both conversions use the
existing FP32 budget as a throughput proxy. The integer counts conservatively
include predicated quotient/remainder corrections and sign restoration; they
are not a measured latency or a total-instruction upper bound.

Known zero divisors, including a zero lane in a constant tensor, are rejected.
Statically visible `INT_MIN / -1` quotient overflow is also rejected; remainder
by `-1` is modeled as zero, including `INT_MIN % -1`. Runtime estimates assume
valid operands; the model does not prove all runtime preconditions. It does not
model register/constant moves, sharing between adjacent quotient/remainder ops,
known-positive operand ranges, exact-division flags, or uniform-warp execution
savings. Nonuniform constant tensors conservatively use the runtime mix.

Other widths and targets retain explicit null typed weights such as
`arith.divsi64`, `arith.remsi64`, or (outside Ampere) `arith.divsi32` /
`arith.remsi32`. Index-typed division is rejected rather than assuming an i32
width. Unsigned and ceiling operations remain uncalibrated. The old
untyped `arith.divsi` and `arith.remsi` configuration weights have been removed.

### Memory and reduction work

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
Target-specific packing/redux instructions and the reduction's internal
inter-warp stage are not modeled yet; that stage is an explicit zero-cost stub.
Explicit `ttg.convert_layout` operations use the separate model above.

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
- Unmodeled integer widths/targets, unsigned/ceiling division/remainder,
  floating-division variants, and external asin weights are null until explicitly
  calibrated. Runs may therefore report more skips than older versions that
  assigned them placeholder weight 1.
- Replace `triton_gpu.convert_layout_cost` with the five `triton_gpu.layout.*`
  weights above. Regenerate equations; the old FP32 per-element proxy is removed.
- Prediction JSON no longer includes the legacy `expression = Max(work...)`
  field, which compared incompatible units. Use `category_expressions` instead.

## Tests

Run the test suite from `analysis/`:

```bash
uv run pytest -q
```
