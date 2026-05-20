#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

class CostIRBuilder {
public:
    CostIRBuilder(MLIRContext *ctx);

    Value zero();
    Value constantCost(double value);
    Value addArgument(llvm::StringRef name, Type type);
    Value addCostArgument(llvm::StringRef name);
    Value add(Value lhs, Value rhs);
    Value sub(Value lhs, Value rhs);
    Value mul(Value lhs, Value rhs);
    Value indexConstant(int64_t value);
    Value indexToCost(Value value);
    Value sumCosts(llvm::ArrayRef<Value> costs);
    void finalize(Value result);
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
