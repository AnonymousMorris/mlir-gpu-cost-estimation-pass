#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>

void analyze_cost(mlir::Operation &op, llvm::raw_ostream &os);
