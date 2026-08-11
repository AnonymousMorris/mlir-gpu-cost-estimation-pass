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

// Return the whole-byte width of a scalar integer or floating-point type.
int64_t type_byte_width(mlir::Type type);

// Return the byte width of the scalar or per-thread TTGIR tensor value.
// Tensor elements that are pointers use the pointee's byte width.
int64_t vector_byte_width(mlir::Value value);
