#include "CostIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include <cassert>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace {

constexpr std::array<llvm::StringLiteral, CostTypeCount> CostTypeNames = {
    "fp32", "fp64", "sfu", "tensor", "l1", "memory"};

} // namespace

CostIRBuilder::CostIRBuilder(MLIRContext *ctx)
    : builder(ctx),
      loc(UnknownLoc::get(ctx)),
      costType(builder.getF64Type()) {
    ctx->loadDialect<func::FuncDialect>();
    ctx->loadDialect<arith::ArithDialect>();
          
    ownedModule = ModuleOp::create(loc);
    module = *ownedModule;

    builder.setInsertionPointToStart(module.getBody());

    llvm::SmallVector<Type, CostTypeCount> resultTypes(CostTypeCount, costType);
    auto funcType = builder.getFunctionType({}, resultTypes);
    costFunc = func::FuncOp::create(builder, loc, "__cost_expr", funcType);

    llvm::SmallVector<DictionaryAttr, CostTypeCount> resultAttrs;
    for (llvm::StringRef name : CostTypeNames) {
        resultAttrs.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr("cost.name", builder.getStringAttr(name))}));
    }
    costFunc.setAllResultAttrs(resultAttrs);

    entry = costFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);
}

Value CostIRBuilder::zero() {
    return arith::getZeroConstant(builder, loc, costType);
}

CostVector CostIRBuilder::zeroVector() {
    Value zero = this->zero();
    CostVector costs;
    costs.fill(zero);
    return costs;
}

CostVector CostIRBuilder::costVector(CostType type, Value cost) {
    CostVector costs = zeroVector();
    costs[static_cast<size_t>(type)] = cost;
    return costs;
}

Value CostIRBuilder::constantCost(double value) {
    return arith::ConstantFloatOp::create(builder, loc, cast<FloatType>(costType),
                                         llvm::APFloat(value));
}

Value CostIRBuilder::addArgument(llvm::StringRef name, Type type,
                                 llvm::StringRef kind) {
    auto argumentIt = arguments.find(name);
    if (argumentIt != arguments.end()) {
        assert(argumentIt->second.getType() == type && "argument type mismatch");
        assert(argumentKinds.lookup(name) == kind && "argument kind mismatch");
        return argumentIt->second;
    }

    auto nameLoc = NameLoc::get(builder.getStringAttr(name), loc);
    Value argument = entry->addArgument(type, nameLoc);

    llvm::SmallVector<Type> inputs(entry->getArgumentTypes());
    auto newFuncType =
        builder.getFunctionType(inputs, costFunc.getResultTypes());
    costFunc.setFunctionType(newFuncType);

    llvm::SmallVector<DictionaryAttr> argAttrs;
    if (ArrayAttr existingArgAttrs = costFunc.getArgAttrsAttr()) {
        for (Attribute attr : existingArgAttrs) {
            argAttrs.push_back(cast<DictionaryAttr>(attr));
        }
    }
    while (argAttrs.size() < costFunc.getNumArguments()) {
        argAttrs.push_back(builder.getDictionaryAttr({}));
    }
    argAttrs[cast<BlockArgument>(argument).getArgNumber()] =
        builder.getDictionaryAttr({
            builder.getNamedAttr("cost.kind", builder.getStringAttr(kind)),
            builder.getNamedAttr("cost.name", builder.getStringAttr(name)),
        });
    costFunc.setAllArgAttrs(argAttrs);

    arguments.try_emplace(name, argument);
    argumentKinds.try_emplace(name, kind);
    return argument;
}

Value CostIRBuilder::addRuntimeArgument(llvm::StringRef name, Type type) {
    return addArgument(name, type, "runtime");
}

Value CostIRBuilder::addCostArgument(llvm::StringRef name) {
    return addArgument(name, costType, "weight");
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

CostVector CostIRBuilder::add(CostVector lhsVector, CostVector rhsVector) {
    for (size_t idx = 0; idx < CostTypeCount; idx++) {
        Value &lhs = lhsVector[idx];
        Value &rhs = rhsVector[idx];
        rhs = add(lhs, rhs);
    }
    return rhsVector;
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

CostVector CostIRBuilder::mul(CostVector lhsVector, Value rhs) {
    for (Value &lhs : lhsVector) {
        if (!mlir::matchPattern(lhs, m_AnyZeroFloat())) {
            lhs = mul(lhs, rhs);
        }
    }
    return lhsVector;
}

Value CostIRBuilder::max(Value lhs, Value rhs) {
    assert(lhs.getType() == rhs.getType());
    return arith::MaximumFOp::create(builder, loc, lhs, rhs);
}

CostVector CostIRBuilder::max(CostVector lhsVector, CostVector rhsVector) {
    for (size_t idx = 0; idx < CostTypeCount; idx++) {
        rhsVector[idx] = max(lhsVector[idx], rhsVector[idx]);
    }
    return rhsVector;
}

Value CostIRBuilder::indexConstant(int64_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value);
}

Value CostIRBuilder::indexToCost(Value value) {
    Type i64Type = builder.getI64Type();
    Value asI64 = value;

    Type valueType = value.getType();
    if (isa<IndexType>(valueType)) {
        asI64 = arith::IndexCastUIOp::create(builder, loc, i64Type, value);
    } 
    else if (auto intType = dyn_cast<IntegerType>(valueType)) {
        if (intType.getWidth() < 64) {
            asI64 = arith::ExtUIOp::create(builder, loc, i64Type, value);
        } else if (intType.getWidth() > 64) {
            asI64 = arith::TruncIOp::create(builder, loc, i64Type, value);
        }
    } else {
        llvm::report_fatal_error("expected integer or index cost value");
    }
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

CostVector CostIRBuilder::sumCosts(llvm::ArrayRef<CostVector> costs) {
    CostVector sum = zeroVector();

    for (CostVector cost : costs) {
        sum = add(sum, cost);
    }

    return sum;
}

void CostIRBuilder::finalize(CostVector resultVec) {
    builder.setInsertionPointToEnd(entry);
    func::ReturnOp::create(builder, loc, resultVec);
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
