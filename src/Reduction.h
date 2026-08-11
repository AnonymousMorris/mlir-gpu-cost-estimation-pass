#pragma once

#include "CostIRBuilder.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

struct GpuSpec;

CostVector analyze_triton_reduce(CostIRBuilder &costBuilder,
                                 mlir::triton::ReduceOp reduceOp,
                                 const GpuSpec &gpu);
