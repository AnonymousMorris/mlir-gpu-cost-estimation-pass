#include "MemoryMask.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <optional>
#include <string>

using namespace mlir;

namespace {

// These operations repeat/reorder logical values without changing the fraction
// of true elements. Do not follow arithmetic, selects, loops, or program IDs.
Value strip_views(Value value) {
    while (Operation *op = value.getDefiningOp()) {
        if (!isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
                 triton::gpu::ConvertLayoutOp>(op)) {
            break;
        }
        value = op->getOperand(0);
    }
    return value;
}

std::optional<llvm::APInt> constant_integer(Value value) {
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    if (!constant) {
        return {};
    }
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue())) {
        return integer.getValue();
    }
    if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
        dense && dense.isSplat()) {
        return dense.getSplatValue<llvm::APInt>();
    }
    return {};
}

// Only entry arguments can be bound from scalar_args. In particular, a loop
// induction variable or carried value must not become a fake host argument.
bool is_kernel_argument(Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    return argument && value.getType().isInteger(32) &&
           isa<triton::FuncOp>(argument.getOwner()->getParentOp());
}

Value wide_bound(CostIRBuilder &costBuilder, Value bound, bool isUnsigned) {
    OpBuilder &builder = costBuilder.getBuilder();
    Location loc = costBuilder.getLoc();
    if (auto constant = constant_integer(bound)) {
        int64_t value = isUnsigned ? constant->getZExtValue()
                                   : constant->getSExtValue();
        return arith::ConstantOp::create(builder, loc,
                                         builder.getI64IntegerAttr(value));
    }
    auto argument = cast<BlockArgument>(bound);
    auto nameLoc = argument.getLoc()->findInstanceOf<NameLoc>();
    std::string name = nameLoc ? nameLoc.getName().str()
                               : "arg" + std::to_string(argument.getArgNumber());
    Value runtime = costBuilder.addRuntimeArgument(name, bound.getType());
    if (isUnsigned) {
        return arith::ExtUIOp::create(builder, loc, builder.getI64Type(), runtime);
    }
    return arith::ExtSIOp::create(builder, loc, builder.getI64Type(), runtime);
}

std::optional<Value> range_fraction(CostIRBuilder &costBuilder,
                                    arith::CmpIOp comparison) {
    Value rangeValue = strip_views(comparison.getLhs());
    Value bound = strip_views(comparison.getRhs());
    auto predicate = comparison.getPredicate();
    if (!rangeValue.getDefiningOp<triton::MakeRangeOp>()) {
        std::swap(rangeValue, bound);
        using P = arith::CmpIPredicate;
        switch (predicate) {
        case P::slt: predicate = P::sgt; break;
        case P::sle: predicate = P::sge; break;
        case P::sgt: predicate = P::slt; break;
        case P::sge: predicate = P::sle; break;
        case P::ult: predicate = P::ugt; break;
        case P::ule: predicate = P::uge; break;
        case P::ugt: predicate = P::ult; break;
        case P::uge: predicate = P::ule; break;
        case P::eq:
        case P::ne: break;
        }
    }
    auto range = rangeValue.getDefiningOp<triton::MakeRangeOp>();
    if (!range || !getElementTypeOrSelf(bound.getType()).isInteger(32) ||
        (!constant_integer(bound) && !is_kernel_argument(bound))) {
        return {};
    }
    bool isUnsigned = predicate == arith::CmpIPredicate::ult ||
                      predicate == arith::CmpIPredicate::ule ||
                      predicate == arith::CmpIPredicate::ugt ||
                      predicate == arith::CmpIPredicate::uge;
    int64_t start = range.getStart();
    int64_t length = int64_t(range.getEnd()) - start;
    if (length <= 0 || (isUnsigned && start < 0)) {
        return {};
    }
    // make_range has i32 elements. Widen before subtraction and the inclusive
    // endpoint adjustment, so INT_MIN/INT_MAX bounds cannot wrap the count.
    OpBuilder &builder = costBuilder.getBuilder();
    Location loc = costBuilder.getLoc();
    auto integer = [&](int64_t value) -> Value {
        return arith::ConstantOp::create(builder, loc,
                                         builder.getI64IntegerAttr(value));
    };
    Value zero = integer(0);
    Value size = integer(length);
    Value difference = arith::SubIOp::create(
        builder, loc, wide_bound(costBuilder, bound, isUnsigned), integer(start));
    auto clamp = [&](Value value) -> Value {
        Value nonnegative = arith::MaxSIOp::create(builder, loc, value, zero);
        return arith::MinSIOp::create(builder, loc, nonnegative, size);
    };
    Value less = clamp(difference);
    Value lessEqual = clamp(arith::AddIOp::create(
        builder, loc, difference, integer(1)));
    Value active;
    switch (predicate) {
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult: active = less; break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule: active = lessEqual; break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
        active = arith::SubIOp::create(builder, loc, size, lessEqual); break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
        active = arith::SubIOp::create(builder, loc, size, less); break;
    case arith::CmpIPredicate::eq:
        active = arith::SubIOp::create(builder, loc, lessEqual, less); break;
    case arith::CmpIPredicate::ne: {
        Value equal = arith::SubIOp::create(builder, loc, lessEqual, less);
        active = arith::SubIOp::create(builder, loc, size, equal); break;
    }
    }
    return costBuilder.mul(costBuilder.indexToCost(active),
                            costBuilder.constantCost(1.0 / length));
}

} // namespace

void record_memory_mask_upper_bound(CostIRBuilder &costBuilder,
                                    Operation &access, llvm::StringRef reason) {
    OpBuilder &builder = costBuilder.getBuilder();
    auto function = costBuilder.getCost();
    constexpr llvm::StringLiteral attribute = "cost.memory_mask_approximations";
    llvm::SmallVector<Attribute> notes;
    if (auto existing = function->getAttrOfType<ArrayAttr>(attribute)) {
        notes.append(existing.begin(), existing.end());
    }
    auto note = builder.getStringAttr(access.getName().getStringRef().str() +
                                      ": " + reason.str());
    if (!llvm::is_contained(notes, note)) {
        notes.push_back(note);
    }
    function->setAttr(attribute, builder.getArrayAttr(notes));
}

Value memory_mask_fraction(CostIRBuilder &costBuilder, Value mask,
                           Operation &access) {
    if (!mask) {
        return costBuilder.constantCost(1);
    }
    mask = strip_views(mask);
    if (auto scalar = constant_integer(mask)) {
        return costBuilder.constantCost(scalar->isZero() ? 0 : 1);
    }
    if (auto constant = mask.getDefiningOp<arith::ConstantOp>()) {
        if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue())) {
            int64_t active = 0;
            for (const auto &value : dense.getValues<llvm::APInt>()) {
                active += !value.isZero();
            }
            return costBuilder.constantCost(
                double(active) / dense.getNumElements());
        }
    }
    if (auto comparison = mask.getDefiningOp<arith::CmpIOp>()) {
        if (auto fraction = range_fraction(costBuilder, comparison)) {
            return *fraction;
        }
    }
    record_memory_mask_upper_bound(
        costBuilder, access,
        "unsupported mask; using full payload (only constants and direct "
        "range comparisons with constant/kernel-argument bounds are modeled)");
    return costBuilder.constantCost(1);
}
