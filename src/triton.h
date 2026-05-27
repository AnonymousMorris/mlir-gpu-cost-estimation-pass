#pragma once

#include "mlir/IR/Value.h"

#include <optional>

class CostIRBuilder;
struct GpuSpec;

std::optional<mlir::Value> analyze_triton_tensor_op(CostIRBuilder &costBuilder,
                                                    mlir::Operation &op,
                                                    const GpuSpec &gpu);

int64_t elements_per_thread(mlir::Operation &op);
int64_t elements_per_thread(mlir::Value value);
