#include "CostConfig.h"
#include "CostIRBuilder.h"
#include "Loop.h"
#include "core.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "triton.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include "gpuSpec.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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

CostVector analyze_BB(CostIRBuilder &costBuilder, Block &BB,
                      const GpuSpec &gpu);
CostVector analyze_region(CostIRBuilder &costBuilder, Region &region,
                          const GpuSpec &gpu);
CostVector analyze_memory_op(CostIRBuilder &costBuilder, Operation &op);
std::optional<CostVector> analyze_op(CostIRBuilder &costBuilder, Operation &op,
                                     const GpuSpec &gpu);
std::optional<CostVector> analyze_simple_op(CostIRBuilder &costBuilder,
                                            Operation &op, const GpuSpec &gpu);
static bool hasTensorType(Operation &op);
static bool hasNamedTensorCost(Operation &op);
std::optional<CostVector> analyze_tensor_op(CostIRBuilder &costBuilder,
                                            Operation &op, const GpuSpec &gpu);
static int getCostBitWidth(Operation &op);
static bool needsCostBitWidth(llvm::StringRef costName);

CostVector analyze_function(CostIRBuilder &costBuilder, Operation &op,
                            const GpuSpec &gpu) {
    llvm::SmallVector<CostVector, 10> regionCosts;

    for (Region &region : op.getRegions()) {
        regionCosts.push_back(analyze_region(costBuilder, region, gpu));
    }

    return costBuilder.sumCosts(regionCosts);
}

void analyze_cost(Operation &op, llvm::raw_ostream &os, const GpuSpec &gpu) {
    CostIRBuilder costBuilder(op.getContext());
    llvm::SmallVector<CostVector, 10> regionCosts;

    for (Region &region : op.getRegions()) {
        regionCosts.push_back(analyze_region(costBuilder, region, gpu));
    }

    CostVector cost = costBuilder.sumCosts(regionCosts);
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

CostVector analyze_region(CostIRBuilder &costBuilder, Region &region,
                          const GpuSpec &gpu) {
    llvm::SmallVector<CostVector, 10> blockCosts;
    for (Block &block : region) {
        blockCosts.push_back(analyze_BB(costBuilder, block, gpu));
    }
    return costBuilder.sumCosts(blockCosts);
}

CostVector analyze_BB(CostIRBuilder &costBuilder, Block &BB,
                      const GpuSpec &gpu) {
    llvm::SmallVector<CostVector, 10> opCosts;
    // iterate over ops in basic block and sum
    for (auto &op : BB) {
        auto cost = analyze_op(costBuilder, op, gpu);
        if (cost) {
            opCosts.push_back(cost.value());
        }
    }
    return costBuilder.sumCosts(opCosts);
}

std::optional<CostVector> analyze_op(CostIRBuilder &costBuilder, Operation &op,
                                     const GpuSpec &gpu) {
    // SCF Loop Op
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return analyze_for_op(costBuilder, forOp, gpu);
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        CostVector then_cost =
            analyze_region(costBuilder, ifOp.getThenRegion(), gpu);
        CostVector else_cost =
            analyze_region(costBuilder, ifOp.getElseRegion(), gpu);
        return costBuilder.max(then_cost, else_cost);
    }

    if (isa<scf::WhileOp>(op)) {
        // TODO: Do proper analysis later. 
        exit(-1);
    }
    
    // Func Call Op
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
        Operation *callee = SymbolTable::lookupNearestSymbolFrom(callOp.getOperation(), callOp.getCalleeAttr());
        return analyze_function(costBuilder, *callee, gpu);
    }

    // GPU Kernel Launch Op
    if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
        Operation *callee = SymbolTable::lookupNearestSymbolFrom(launchOp.getOperation(), launchOp.getKernelAttr());
        return analyze_function(costBuilder, *callee, gpu);
    }

    // GPU Tensor/TTGIR op
    if (hasTensorType(op) || hasNamedTensorCost(op)) {
        return analyze_tensor_op(costBuilder, op, gpu);
    }

    // Other Basic Ops with cost defined in costConfig.h
    if (auto cost = analyze_simple_op(costBuilder, op, gpu)) {
        return cost;
    }

    op.emitError("unknown op in cost analysis: ")
        << op.getName().getStringRef();
    llvm::report_fatal_error("unknown op in cost analysis");
    // exit(-1);
}

bool hasTensorType(Operation &op) {
    auto isTensor = [](Value value) {
        return isa<RankedTensorType>(value.getType());
    };

    return llvm::any_of(op.getOperands(), isTensor) ||
        llvm::any_of(op.getResults(), isTensor);
}

bool hasNamedTensorCost(Operation &op) {
    return NamedTensorOpCost.contains(op.getName().getStringRef());
}

int getCostBitWidth(Operation &op) {
    Type type = op.getResult(0).getType();
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
        type = shapedType.getElementType();
    }
    return type.getIntOrFloatBitWidth();
}

bool needsCostBitWidth(llvm::StringRef costName) {
    return costName == "arith.addf" || costName == "arith.subf" ||
           costName == "arith.mulf" || costName == "arith.fma";
}

std::optional<CostVector> analyze_tensor_op(CostIRBuilder &costBuilder,
                                            Operation &op, const GpuSpec &gpu) {
    if (auto simpleValue = analyze_simple_op(costBuilder, op, gpu)) {
        int64_t serialization = elements_per_thread(op);
        return costBuilder.mul(*simpleValue,
                               costBuilder.constantCost(serialization));
    }

    return analyze_triton_tensor_op(costBuilder, op, gpu);
}

// return cost for ops with cost defined in costConfig. 
std::optional<CostVector> analyze_simple_op(CostIRBuilder &costBuilder,
                                            Operation &op, const GpuSpec &gpu) {
    auto simpleCostIt = SimpleOpCosts.find(op.getName().getStringRef());
    auto namedCostIt = NamedOpCosts.find(op.getName().getStringRef());

    // Return constant cost for operation
    if (simpleCostIt != SimpleOpCosts.end()) {
        const CostSpec &cost = simpleCostIt->second;
        if (const auto *constantCost = std::get_if<double>(&cost)) {
            return costBuilder.costVector(CostType::FP32,
                                          costBuilder.constantCost(*constantCost));
        }
    }

    // Return an named variable cost for operation
    if (namedCostIt != NamedOpCosts.end()) {
        const CostSpec &cost = namedCostIt->second;
        const llvm::StringRef costName = std::get<llvm::StringRef>(cost);
        if (needsCostBitWidth(costName)) {
            int bitWidth = getCostBitWidth(op);
            std::string costNameWithBits =
                costName.str() + std::to_string(bitWidth);
            CostType type = bitWidth == 64 ? CostType::FP64 : CostType::FP32;
            return costBuilder.costVector(
                type, costBuilder.addCostArgument(costNameWithBits));
        }
        return costBuilder.costVector(CostType::FP32,
                                      costBuilder.addCostArgument(costName));
    }

    if (auto loadOp = dyn_cast<xegpu::LoadNdOp>(op)) {
        return analyze_memory_op(costBuilder, *loadOp.getOperation());
    }
    // No cost is set for the operation
    return {};
}

// Cost = 
// p1 * L1_cost + 
// (1 - p1) * p2 * L2_cost + 
// (1 - p1) * (1 - p2) * p3 * L3_cost + 
// (1 - p1) * (1 - p2) * (1 - p3) * Global_cost
CostVector analyze_memory_op(CostIRBuilder &costBuilder, Operation &op) {
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
    Value global_prob = costBuilder.mul(l3_pre, miss3);
    Value global_term = costBuilder.mul(global_prob, global_cost);

    llvm::SmallVector<Value, 4> terms;
    terms.push_back(l1_term);
    terms.push_back(l2_term);
    terms.push_back(l3_term);
    terms.push_back(global_term);

    return costBuilder.costVector(CostType::MEMORY, costBuilder.sumCosts(terms));
}
