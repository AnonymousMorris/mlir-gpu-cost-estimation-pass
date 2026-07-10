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

enum class CostType {
    FP32,
    FP64,
    SFU,
    TENSOR,
    MEMORY,

    count,
};

constexpr size_t CostTypeCount = static_cast<size_t>(CostType::count);
using CostVector = std::array<Value, CostTypeCount>;

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
};
