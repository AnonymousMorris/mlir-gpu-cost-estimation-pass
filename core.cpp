#include "CostConfig.h"
#include "CostIRBuilder.h"
#include "Loop.h"
#include "core.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <cassert>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <variant>

using namespace mlir;

Value analyze_BB(CostIRBuilder &costBuilder, Block &BB);
Value analyze_op(CostIRBuilder &costBuilder, Operation &op);
Value analyze_region(CostIRBuilder &costBuilder, Region &region);
Value analyze_simple_op(CostIRBuilder &costBuilder, Operation &op);

void analyze_cost(Operation &op, llvm::raw_ostream &os) {
    CostIRBuilder costBuilder(op.getContext());
    llvm::SmallVector<Value, 10> regionCosts;

    for (Region &region : op.getRegions()) {
        regionCosts.push_back(analyze_region(costBuilder, region));
    }

    Value cost = costBuilder.sumCosts(regionCosts);
    costBuilder.finalize(cost);

    os << "\n// Cost expression for " << op.getName();
    if (auto symName = op.getAttrOfType<StringAttr>("sym_name")) {
        os << " @" << symName.getValue();
    }
    os << "\n";
    costBuilder.getModule().print(os);
    os << "\n";
}

Value analyze_region(CostIRBuilder &costBuilder, Region &region) {
    llvm::SmallVector<Value, 10> blockCosts;
    for (Block &block : region) {
        blockCosts.push_back(analyze_BB(costBuilder, block));
    }
    return costBuilder.sumCosts(blockCosts);
}

Value analyze_BB(CostIRBuilder &costBuilder, Block &BB) {
    llvm::SmallVector<Value, 10> opCosts;
    // iterate over ops in basic block and sum
    for (auto &op : BB) {
        opCosts.push_back(analyze_op(costBuilder, op));
    }
    return costBuilder.sumCosts(opCosts);
}

Value analyze_op(CostIRBuilder &costBuilder, Operation &op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return analyze_for_op(costBuilder, forOp);
    }

    return analyze_simple_op(costBuilder, op);
}

// return cost for ops with cost defined in costConfig. 
Value analyze_simple_op(CostIRBuilder &costBuilder, Operation &op) {
    auto simpleCostIt = SimpleOpCosts.find(op.getName().getStringRef());
    auto namedCostIt = NamedOpCosts.find(op.getName().getStringRef());

    // Return constant cost for operation
    if (simpleCostIt != SimpleOpCosts.end()) {
        const CostSpec &cost = simpleCostIt->second;
        if (const auto *constantCost = std::get_if<double>(&cost)) {
            return costBuilder.constantCost(*constantCost);
        }
    }

    // Return an named variable cost for operation
    if (namedCostIt != NamedOpCosts.end()) {
        std::cerr <<"dbg";
        const CostSpec &cost = namedCostIt->second;
        return costBuilder.addCostArgument(std::get<llvm::StringRef>(cost));
    }

    // No cost is set for the operation, defaulting to zero for now
    std::cerr<< "dbg2";
    return costBuilder.zero();
}
