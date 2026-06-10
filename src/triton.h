#pragma once

#include "CostIRBuilder.h"

#include "mlir/IR/Value.h"

#include <optional>

struct GpuSpec;

std::optional<CostVector> analyze_triton_tensor_op(CostIRBuilder &costBuilder,
                                                   mlir::Operation &op,
                                                   const GpuSpec &gpu);

int64_t elements_per_thread(mlir::Operation &op);
int64_t elements_per_thread(mlir::Value value);
