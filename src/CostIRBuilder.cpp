#include "CostIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include <cassert>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

CostIRBuilder::CostIRBuilder(MLIRContext *ctx)
    : builder(ctx),
      loc(UnknownLoc::get(ctx)),
      costType(builder.getF64Type()) {
    ownedModule = ModuleOp::create(loc);
    module = *ownedModule;

    builder.setInsertionPointToStart(module.getBody());

    auto funcType = builder.getFunctionType({}, {costType});
    costFunc = func::FuncOp::create(builder, loc, "__cost_expr", funcType);

    entry = costFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);
}

Value CostIRBuilder::zero() {
    return arith::getZeroConstant(builder, loc, costType);
}

Value CostIRBuilder::constantCost(double value) {
    return arith::ConstantFloatOp::create(builder, loc, cast<FloatType>(costType),
                                         llvm::APFloat(value));
}

Value CostIRBuilder::addArgument(llvm::StringRef name, Type type) {
    auto argumentIt = arguments.find(name);
    if (argumentIt != arguments.end()) {
        assert(argumentIt->second.getType() == type && "argument type mismatch");
        return argumentIt->second;
    }

    auto nameLoc = NameLoc::get(builder.getStringAttr(name), loc);
    Value argument = entry->addArgument(type, nameLoc);

    llvm::SmallVector<Type> inputs(entry->getArgumentTypes());
    auto newFuncType = builder.getFunctionType(inputs, {costType});
    costFunc.setFunctionType(newFuncType);

    arguments.try_emplace(name, argument);
    return argument;
}

Value CostIRBuilder::addCostArgument(llvm::StringRef name) {
    return addArgument(name, costType);
}

Value CostIRBuilder::add(Value lhs, Value rhs) {
    Type type = lhs.getType();
    assert(type == rhs.getType() && "add operands must have the same type");

    if (isa<FloatType>(type)) {
        return arith::AddFOp::create(builder, loc, lhs, rhs);
    }

    assert(type.isIntOrIndex() && "unsupported add operand type");
    return arith::AddIOp::create(builder, loc, lhs, rhs);
}

Value CostIRBuilder::sub(Value lhs, Value rhs) {
    Type type = lhs.getType();
    assert(type == rhs.getType() && "add operands must have the same type");

    if (isa<FloatType>(type)) {
        return arith::SubFOp::create(builder, loc, lhs, rhs);
    }

    assert(type.isIntOrIndex() && "unsupported add operand type");
    return arith::SubIOp::create(builder, loc, lhs, rhs);
}

Value CostIRBuilder::mul(Value lhs, Value rhs) {
    Type type = lhs.getType();
    assert(type == rhs.getType() && "mul operands must have the same type");

    if (isa<FloatType>(type)) {
        return arith::MulFOp::create(builder, loc, lhs, rhs);
    }

    assert(type.isIntOrIndex() && "unsupported mul operand type");
    return arith::MulIOp::create(builder, loc, lhs, rhs);
}

Value CostIRBuilder::indexConstant(int64_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value);
}

Value CostIRBuilder::indexToCost(Value value) {
    Value asI64 = arith::IndexCastUIOp::create(builder, loc, builder.getI64Type(),
                                              value);
    return arith::UIToFPOp::create(builder, loc, costType, asI64);
}

Value CostIRBuilder::sumCosts(llvm::ArrayRef<Value> costs) {
    Value sum = zero();

    for (Value cost : costs) {
        assert(cost.getType() == costType && "cost type mismatch");
        sum = add(sum, cost);
    }

    return sum;
}

void CostIRBuilder::finalize(Value result) {
    builder.setInsertionPointToEnd(entry);
    func::ReturnOp::create(builder, loc, result);
}

void CostIRBuilder::simplify() {
    PassManager pm(module->getContext());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    if (failed(pm.run(module))) {
        llvm::report_fatal_error("failed to simplify cost expression");
    }
}

ModuleOp CostIRBuilder::getModule() {
    return module;
}

func::FuncOp CostIRBuilder::getCost() {
    return costFunc;
}

OpBuilder &CostIRBuilder::getBuilder() {
    return builder;
}

Location CostIRBuilder::getLoc() const {
    return loc;
}
