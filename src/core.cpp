#include "CostConfig.h"
#include "CostIRBuilder.h"
#include "Loop.h"
#include "core.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <variant>
#include <cassert>

using namespace mlir;

Value analyze_BB(CostIRBuilder &costBuilder, Block &BB);
Value analyze_region(CostIRBuilder &costBuilder, Region &region);
std::optional<Value> analyze_op(CostIRBuilder &costBuilder, Operation &op);
std::optional<Value> analyze_simple_op(CostIRBuilder &costBuilder, Operation &op);

Value analyze_function(CostIRBuilder &costBuilder, Operation &op) {
    llvm::SmallVector<Value, 10> regionCosts;

    for (Region &region : op.getRegions()) {
        regionCosts.push_back(analyze_region(costBuilder, region));
    }

    Value cost = costBuilder.sumCosts(regionCosts);
    return cost;
}

void analyze_cost(Operation &op, llvm::raw_ostream &os) {
    CostIRBuilder costBuilder(op.getContext());
    llvm::SmallVector<Value, 10> regionCosts;

    for (Region &region : op.getRegions()) {
        regionCosts.push_back(analyze_region(costBuilder, region));
    }

    Value cost = costBuilder.sumCosts(regionCosts);
    costBuilder.finalize(cost);
    costBuilder.simplify();

    os << "\n// Cost expression for " << op.getName();
    if (auto symName = op.getAttrOfType<StringAttr>("sym_name")) {
        os << " @" << symName.getValue();
    }
    os << "\n";
    OpPrintingFlags flags;
    flags.printNameLocAsPrefix();
    costBuilder.getCost().print(os, flags);
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
        auto cost = analyze_op(costBuilder, op);
        if (cost) {
            opCosts.push_back(cost.value());
        }
    }
    return costBuilder.sumCosts(opCosts);
}

std::optional<Value> analyze_op(CostIRBuilder &costBuilder, Operation &op) {
    // SCF Loop Op
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return analyze_for_op(costBuilder, forOp);
    }
    
    // Func Call Op
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
        Operation *callee = SymbolTable::lookupNearestSymbolFrom(callOp.getOperation(), callOp.getCalleeAttr());
        return analyze_function(costBuilder, *callee);
    }

    // GPU Kernel Launch Op
    if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
        Operation *callee = SymbolTable::lookupNearestSymbolFrom(launchOp.getOperation(), launchOp.getKernelAttr());
        return analyze_function(costBuilder, *callee);
    }

    // Other Basic Ops with cost defined in costConfig.h
    return analyze_simple_op(costBuilder, op);
}

// return cost for ops with cost defined in costConfig. 
std::optional<Value> analyze_simple_op(CostIRBuilder &costBuilder, Operation &op) {
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
        const CostSpec &cost = namedCostIt->second;
        return costBuilder.addCostArgument(std::get<llvm::StringRef>(cost));
    }

    // No cost is set for the operation
    return {};
}

// Cost = 
// p1 * L1_cost + 
// (1 - p1) * p2 * L2_cost + 
// (1 - p1) * (1 - p2) * p3 * L3_cost + 
// (1 - p1) * (1 - p2) * (1 - p3) * Global_cost
Value analyze_memory_op(CostIRBuilder &costBuilder, Operation &op) {
    Value one = costBuilder.constantCost(1.0);
    Value p1 = costBuilder.addCostArgument("residency.p1");
    Value l1_cost = costBuilder.addCostArgument("residency.l1_cost");
    Value p2 = costBuilder.addCostArgument("residency.p2");
    Value l2_cost = costBuilder.addCostArgument("residency.l2_cost");
    Value p3 = costBuilder.addCostArgument("residency.p3");
    Value l3_cost = costBuilder.addCostArgument("residency.l3_cost");
    Value global_cost = costBuilder.addCostArgument("residency.global_cost");

    Value miss1 = costBuilder.sub(one, p1);
    Value miss2 = costBuilder.sub(one, p2);
    Value miss3 = costBuilder.sub(one, p3);

    Value l1_term = costBuilder.mul(p1, l1_cost);
    Value l2_prob = costBuilder.mul(miss1, p2);
    Value l2_term = costBuilder.mul(l2_prob, l2_cost);
    Value l3_pre = costBuilder.mul(l2_prob, miss2);
    Value l3_prob = costBuilder.mul(l3_pre, p3);
    Value l3_term = costBuilder.mul(l3_prob, l3_cost);
    Value global_prob = costBuilder.mul(l3_prob, miss3);
    Value global_term = costBuilder.mul(global_prob, global_cost);

    llvm::SmallVector<Value, 4> terms;
    terms.push_back(l1_term);
    terms.push_back(l2_term);
    terms.push_back(l3_term);
    terms.push_back(global_term);

    return costBuilder.sumCosts(terms);
}
