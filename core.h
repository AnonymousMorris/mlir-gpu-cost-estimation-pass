#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

void analyze_cost(mlir::Operation &op, llvm::raw_ostream &os);
