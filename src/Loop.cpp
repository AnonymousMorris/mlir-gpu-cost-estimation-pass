#include "Loop.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

#include <cstdlib>

using namespace mlir;

struct GpuSpec;

CostVector analyze_region(CostIRBuilder &costBuilder, Region &region,
                          const GpuSpec &gpu);

namespace {

bool can_clone_bound_op(Operation *op) {
    return isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
               arith::DivSIOp, arith::DivUIOp, arith::CeilDivSIOp,
               arith::CeilDivUIOp, arith::IndexCastOp,
               arith::IndexCastUIOp>(op);
}

std::string source_argument_name(BlockArgument argument) {
    if (auto nameLoc = argument.getLoc()->findInstanceOf<NameLoc>()) {
        return nameLoc.getName().str();
    }
    return "arg" + std::to_string(argument.getArgNumber());
}

FailureOr<Value> clone_bound_expr(CostIRBuilder &costBuilder, Value value,
                                  IRMapping &mapping) {
    if (Value mapped = mapping.lookupOrNull(value)) {
        return mapped;
    }

    if (auto argument = dyn_cast<BlockArgument>(value)) {
        Value mapped = costBuilder.addRuntimeArgument(
            source_argument_name(argument), argument.getType());
        mapping.map(value, mapped);
        return mapped;
    }

    Operation *op = value.getDefiningOp();
    if (!op || op->getNumRegions() != 0 || op->getNumResults() != 1 ||
        !can_clone_bound_op(op)) {
        return failure();
    }

    for (Value operand : op->getOperands()) {
        FailureOr<Value> mappedOperand =
            clone_bound_expr(costBuilder, operand, mapping);
        if (failed(mappedOperand)) {
            return failure();
        }
        mapping.map(operand, *mappedOperand);
    }

    Operation *cloned = costBuilder.getBuilder().clone(*op, mapping);
    Value clonedResult = cloned->getResult(0);
    mapping.map(value, clonedResult);
    return clonedResult;
}

Value build_for_trip_count(CostIRBuilder &costBuilder, scf::ForOp forOp) {
    IRMapping mapping;

    FailureOr<Value> lower =
        clone_bound_expr(costBuilder, forOp.getLowerBound(), mapping);
    FailureOr<Value> upper =
        clone_bound_expr(costBuilder, forOp.getUpperBound(), mapping);
    FailureOr<Value> step =
        clone_bound_expr(costBuilder, forOp.getStep(), mapping);

    if (failed(lower) || failed(upper) || failed(step)) {
        forOp.emitOpError("cannot analyze symbolic loop trip count");
        std::exit(EXIT_FAILURE);
    }

    OpBuilder &builder = costBuilder.getBuilder();
    Location loc = costBuilder.getLoc();

    // Widen before subtracting: even valid bounds can have a difference that
    // overflows their original signed type. Index bounds use the 64-bit GPU
    // index width. The extra bit also lets unsigned bounds be compared signed.
    auto widen = [&](Value value, bool isUnsigned) -> Value {
        if (value.getType().isIndex()) {
            value = arith::IndexCastOp::create(builder, loc,
                                              builder.getI64Type(), value);
        }
        auto type = cast<IntegerType>(value.getType());
        Type wideType = builder.getIntegerType(type.getWidth() + 1);
        if (isUnsigned) {
            return arith::ExtUIOp::create(builder, loc, wideType, value);
        }
        return arith::ExtSIOp::create(builder, loc, wideType, value);
    };
    Value wideLower = widen(*lower, forOp.getUnsignedCmp());
    Value wideUpper = widen(*upper, forOp.getUnsignedCmp());
    Value wideStep = widen(*step, /*isUnsigned=*/true);
    Value range = arith::SubIOp::create(builder, loc, wideUpper, wideLower);
    Value zero = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(range.getType(), 0));
    Value nonnegativeRange = arith::MaxSIOp::create(builder, loc, range, zero);
    return arith::CeilDivUIOp::create(builder, loc, nonnegativeRange, wideStep);
}

} // namespace

CostVector analyze_for_op(CostIRBuilder &costBuilder, scf::ForOp forOp,
                          const GpuSpec &gpu) {
    CostVector bodyCost = analyze_region(costBuilder, forOp.getRegion(), gpu);
    Value tripCount = build_for_trip_count(costBuilder, forOp);
    return costBuilder.mul(bodyCost, costBuilder.indexToCost(tripCount));
}
