#pragma once

#include "CostIRBuilder.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

struct GpuSpec;

// Per-thread movement work. Block scaling is applied by analyze_cost.
CostVector analyze_layout_conversion(
    CostIRBuilder &builder, mlir::triton::gpu::ConvertLayoutOp op,
    const GpuSpec &gpu);
