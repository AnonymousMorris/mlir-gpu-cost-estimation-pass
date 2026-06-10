#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <array>

using namespace mlir;

using CostVector = std::array<Value, 5>;

enum class CostType {
    FP32,
    FP64,
    SFU,
    TENSOR,
    MEMORY,

    count,
};

class CostIRBuilder {
public:
    CostIRBuilder(MLIRContext *ctx);

    Value zero();
    CostVector zeroVector();
    CostVector costVector(CostType type, Value cost);
    Value constantCost(double value);
    Value addArgument(llvm::StringRef name, Type type);
    Value addCostArgument(llvm::StringRef name);
    Value add(Value lhs, Value rhs);
    CostVector add(CostVector lhsVector, CostVector rhsVector);
    Value sub(Value lhs, Value rhs);
    Value mul(Value lhs, Value rhs);
    CostVector mul(CostVector lhsVector, Value rhs);
    Value max(Value lhs, Value rhs);
    CostVector max(CostVector lhsVector, CostVector rhsVector);
    Value indexConstant(int64_t value);
    Value indexToCost(Value value);
    Value sumCosts(llvm::ArrayRef<Value> costs);
    CostVector sumCosts(llvm::ArrayRef<CostVector> costs);
    void finalize(Value result);
    void finalize(CostVector resultVec);
    void simplify();

    ModuleOp getModule();
    func::FuncOp getCost();
    OpBuilder &getBuilder();
    Location getLoc() const;

private:
    OpBuilder builder;
    Location loc;
    Type costType;

    OwningOpRef<ModuleOp> ownedModule;
    ModuleOp module;
    func::FuncOp costFunc;
    Block *entry = nullptr;
    llvm::StringMap<Value> arguments;

    // We want to treat the different pipelines in the GPU
    // independently of one another as they should be able 
    // to execute in parallel
    CostVector typeCosts;
};
