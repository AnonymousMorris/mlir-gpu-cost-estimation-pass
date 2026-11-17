#include "IntegerDivision.h"
#include "gpuSpec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DivisionByConstantInfo.h"
#include <cstdlib>

using namespace mlir;

namespace {

Value unwrapSplat(Value value) {
    if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
        return splat.getSrc();
    }
    return value;
}

std::optional<llvm::APInt> constantInteger(Value value) {
    llvm::APInt result;
    if (matchPattern(unwrapSplat(value), m_ConstantInt(&result))) {
        return result;
    }
    return {};
}

bool containsConstant(Value value, const llvm::APInt &target) {
    Attribute attr;
    if (!matchPattern(unwrapSplat(value), m_Constant(&attr))) {
        return false;
    }
    if (auto scalar = dyn_cast<IntegerAttr>(attr)) {
        return scalar.getValue() == target;
    }
    if (auto tensor = dyn_cast<DenseIntElementsAttr>(attr)) {
        return llvm::any_of(tensor.getValues<llvm::APInt>(),
                            [&](const llvm::APInt &v) { return v == target; });
    }
    return false;
}

bool knownSignedOverflow(Value lhs, Value rhs) {
    if (auto d = constantInteger(rhs)) {
        return d->isAllOnes() &&
               containsConstant(lhs, llvm::APInt::getSignedMinValue(32));
    }
    if (auto n = constantInteger(lhs); n && n->isMinSignedValue()) {
        return containsConstant(rhs, llvm::APInt::getAllOnes(32));
    }
    DenseIntElementsAttr numerators, divisors;
    if (matchPattern(unwrapSplat(lhs), m_Constant(&numerators)) &&
        matchPattern(unwrapSplat(rhs), m_Constant(&divisors))) {
        for (auto [n, d] : llvm::zip(numerators.getValues<llvm::APInt>(),
                                    divisors.getValues<llvm::APInt>())) {
            if (n.isMinSignedValue() && d.isAllOnes()) {
                return true;
            }
        }
    }
    return false;
}

CostVector constantCost(CostIRBuilder &builder, bool remainder, unsigned count) {
    if (count == 0) {
        return builder.zeroVector();
    }
    llvm::StringRef name = remainder ? "arith.remsi32.constant.int32_cost"
                                     : "arith.divsi32.constant.int32_cost";
    return builder.costVector(
        CostType::INT32,
        builder.mul(builder.addCostArgument(name), builder.constantCost(count)));
}

unsigned constantInstructionCount(llvm::APInt divisor, bool remainder) {
    if (divisor.isOne()) {
        return 0;
    }
    if (divisor.isAllOnes()) {
        return remainder ? 0 : 1; // x / -1 is a negation for valid inputs.
    }
    if (divisor.isMinSignedValue() && !remainder) {
        return 2; // Compare x == INT_MIN and materialize its boolean result.
    }

    llvm::APInt magnitude = divisor.abs();
    if (magnitude.isPowerOf2()) {
        // Signed truncation needs bias before shifting, not just an arithmetic
        // right shift. Ampere LEA fuses the bias shift/add.
        unsigned count = magnitude == 2 ? 2 : 3;
        return remainder ? count + 1 : count + divisor.isNegative();
    }

    // srem has the same result for d and -d, so use the positive divisor.
    auto info = llvm::SignedDivisionByConstantInfo::get(
        remainder ? magnitude : divisor);
    // IMAD.HI (including a positive numerator correction) + LEA rounding,
    // plus a shift when needed. A negative numerator correction needs negation.
    unsigned count = 2 + (info.ShiftAmount != 0);
    if (!remainder && divisor.isNegative() && info.Magic.isStrictlyPositive()) {
        ++count;
    }
    // The final x - quotient*d fuses to one IMAD.
    return count + remainder;
}

CostVector runtimeCost(CostIRBuilder &builder, bool remainder) {
    llvm::StringRef intName = remainder ? "arith.remsi32.runtime.int32_cost"
                                        : "arith.divsi32.runtime.int32_cost";
    llvm::StringRef fpName = remainder ? "arith.remsi32.runtime.fp32_cost"
                                       : "arith.divsi32.runtime.fp32_cost";
    llvm::StringRef sfuName = remainder ? "arith.remsi32.runtime.sfu_cost"
                                        : "arith.divsi32.runtime.sfu_cost";
    CostVector cost = builder.costVector(CostType::INT32,
                                         builder.addCostArgument(intName));
    cost = builder.add(cost, builder.costVector(
        CostType::FP32, builder.addCostArgument(fpName)));
    return builder.add(cost, builder.costVector(
        CostType::SFU, builder.addCostArgument(sfuName)));
}

} // namespace

std::optional<CostVector> analyze_signed_integer_division(
    CostIRBuilder &builder, Operation &op, const GpuSpec &gpu) {
    if (!isa<arith::DivSIOp, arith::RemSIOp>(op) ||
        !getElementTypeOrSelf(op.getResult(0).getType()).isInteger(32) ||
        !gpu.isAmpere()) {
        return {};
    }

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto divisor = constantInteger(rhs);
    if (containsConstant(rhs, llvm::APInt(32, 0))) {
        op.emitError("cost analysis cannot model signed division/remainder by zero");
        std::exit(EXIT_FAILURE);
    }
    if (isa<arith::DivSIOp>(op) && knownSignedOverflow(lhs, rhs)) {
        op.emitError("cost analysis cannot model signed division overflow INT_MIN / -1");
        std::exit(EXIT_FAILURE);
    }

    if (divisor) {
        if (constantInteger(lhs)) {
            return builder.zeroVector(); // LLVM folds valid scalar/splat constants.
        }
        bool remainder = isa<arith::RemSIOp>(op);
        return constantCost(builder, remainder,
                            constantInstructionCount(*divisor, remainder));
    }
    // Runtime or nonuniform tensor divisor: conservative valid-input mix.
    return runtimeCost(builder, isa<arith::RemSIOp>(op));
}
