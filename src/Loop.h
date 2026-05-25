#pragma once

#include "CostIRBuilder.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

struct GpuSpec;

mlir::Value analyze_for_op(CostIRBuilder &costBuilder, mlir::scf::ForOp forOp,
                           const GpuSpec &gpu);
