#pragma once

#include "CostIRBuilder.h"

// Fraction of the existing tensor payload enabled by a supported mask.
// Unsupported masks retain fraction 1 and record an explicit upper-bound note.
mlir::Value memory_mask_fraction(CostIRBuilder &costBuilder, mlir::Value mask,
                                 mlir::Operation &access);

void record_memory_mask_upper_bound(CostIRBuilder &costBuilder,
                                    mlir::Operation &access,
                                    llvm::StringRef reason);
