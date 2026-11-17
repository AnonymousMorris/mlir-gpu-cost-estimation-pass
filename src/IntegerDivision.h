#pragma once

#include "CostIRBuilder.h"
#include <optional>

struct GpuSpec;

// Returns per-lane work for modeled Ampere i32 signed division/remainder.
// Tensor and block scaling are applied by the caller. Empty means unmodeled.
std::optional<CostVector> analyze_signed_integer_division(
    CostIRBuilder &costBuilder, mlir::Operation &op, const GpuSpec &gpu);
