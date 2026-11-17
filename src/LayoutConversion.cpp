#include "LayoutConversion.h"
#include "gpuSpec.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

struct MovementWork {
    int64_t shuffles = 0;
    int64_t selects = 0;
    int64_t sharedBytes = 0;
    int64_t warpSyncs = 0;
    int64_t ctaBarriers = 0;
};

[[noreturn]] void unsupported(ConvertLayoutOp op, llvm::StringRef reason) {
    op.emitError("unsupported layout conversion in cost analysis: ") << reason;
    std::exit(EXIT_FAILURE);
}

LinearLayout withoutRegisterBroadcasts(RankedTensorType type) {
    auto layout = toLinearLayout(type);
    return actionRemoveBroadcastedRegs(layout).apply(layout);
}

MovementWork warpWork(RankedTensorType src, RankedTensorType dst, int bitwidth) {
    auto factors = getWarpLayoutConvertDecomposition(src, dst, bitwidth);
    auto kReg = StringAttr::get(src.getContext(), "register");
    auto kLane = StringAttr::get(src.getContext(), "lane");
    int64_t words = factors.pReg.getInDimSize(kReg) /
                    (int64_t{1} << factors.nPack) *
                    llvm::divideCeil(bitwidth, 32);
    int64_t mixed = factors.mixedTranspositions.size();
    bool laneIdentity = squareSublayoutIsIdentity(factors.pLane, kLane);

    // Match the backend's choice between its Ship and Swap implementations.
    auto shippable = [](const auto &t) {
        return t.topPreSel == t.topPostSel ||
               (t.topPreSel == 0x5140 && t.topPostSel == 0x6240) ||
               (t.topPreSel == 0x6420 && t.topPostSel == 0x5410) ||
               (t.topPreSel == 0x3210 && t.topPostSel == 0x3120);
    };
    MovementWork work;
    if (mixed == 1 && laneIdentity &&
        shippable(factors.mixedTranspositions.front())) {
        work.shuffles = words / 2;
        work.selects = 3 * words / 2;
    } else {
        work.shuffles = laneIdentity ? words - words / (int64_t{1} << mixed)
                                     : words;
        work.selects = 2 * mixed * words;
    }
    return work;
}

MovementWork sharedWork(RankedTensorType src, RankedTensorType dst, int bitwidth) {
    // The lowering removes register broadcasts before staging through shared
    // memory, but does not predicate away replicated lanes/warps.
    auto srcLayout = withoutRegisterBroadcasts(src);
    auto dstLayout = withoutRegisterBroadcasts(dst);
    auto kReg = StringAttr::get(src.getContext(), "register");
    auto kWarp = StringAttr::get(src.getContext(), "warp");
    auto kReps = StringAttr::get(src.getContext(), "reps");
    int memoryBitwidth = std::max(bitwidth, 8); // i1 is widened to i8.
    // Match getSrcDstTiles for Ampere: ordinary shared stores, with optional
    // ldmatrix loads (stmatrix is not available until Hopper). The selected
    // instruction tile affects scratch repetitions and synchronization count.
    LocalMemOpTile ordinary{{}, {0, 1, 2}};
    SmallVector<LocalMemOpTile> stores{ordinary}, loads{ordinary};
    if (memoryBitwidth <= 32) {
        loads.push_back({{0, 1}, {2, 3, 4}});
    }
    if (memoryBitwidth == 16) {
        loads.push_back({{2, 3, 4}, {0, 1}});
    }
    auto scratch = optimalSwizzling(
        srcLayout, dstLayout, stores, loads, memoryBitwidth).first;
    int64_t repetitions = scratch.getInDimSize(kReps);

    MovementWork work;
    work.sharedBytes = (int64_t{srcLayout.getInDimSize(kReg)} +
                        dstLayout.getInDimSize(kReg)) * (memoryBitwidth / 8);
    int64_t syncs = 2 * repetitions - 1; // store/load, then scratch reuse.
    if (isCvtDimSync(srcLayout, dstLayout, kWarp)) {
        work.warpSyncs = syncs;
    } else {
        work.ctaBarriers = syncs;
    }
    return work;
}

void addWork(CostIRBuilder &builder, CostVector &cost, CostType category,
             llvm::StringRef parameter, int64_t count) {
    if (count == 0) {
        return;
    }
    Value work = builder.mul(builder.addCostArgument(parameter),
                             builder.constantCost(count));
    cost = builder.add(cost, builder.costVector(category, work));
}

} // namespace

CostVector analyze_layout_conversion(
    CostIRBuilder &builder, ConvertLayoutOp op, const GpuSpec &gpu) {
    auto src = op.getSrc().getType();
    auto dst = op.getType();
    auto *ctx = op.getContext();
    auto kBlock = StringAttr::get(ctx, "block");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kLane = StringAttr::get(ctx, "lane");
    auto conversion = minimalCvtLayout(src, dst);
    auto dims = conversion.getInDimNames();
    bool crossesBlock = llvm::is_contained(dims, kBlock);
    bool crossesWarp = llvm::is_contained(dims, kWarp);
    bool crossesLane = llvm::is_contained(dims, kLane);

    if (crossesBlock) {
        unsupported(op, "cross-CTA/distributed shared-memory transfers");
    }
    if (!crossesWarp && !crossesLane) {
        // Identity and register reordering are SSA rewiring in Triton's lowering.
        return builder.zeroVector();
    }
    if (isGenericLinearEncoding(src.getEncoding()) ||
        isGenericLinearEncoding(dst.getEncoding())) {
        unsupported(op, "generic-linear inter-thread transfers are not modeled");
    }
    if (!gpu.isAmpere()) {
        unsupported(op, "inter-thread transfers require an Ampere target");
    }
    Type element = src.getElementType();
    if (!element.isIntOrFloat() && !isa<triton::PointerType>(element)) {
        unsupported(op, "unmodeled element type");
    }
    int bitwidth = getIntOrFloatOrPtrBitWidth(element);
    if (bitwidth != 1 && bitwidth != 8 && bitwidth != 16 &&
        bitwidth != 32 && bitwidth != 64) {
        unsupported(op, "unmodeled element bit width");
    }
    bool forceShuffle =
        op->getParentOp()->hasAttrOfType<UnitAttr>("always_use_warp_shuffle");
    if (forceShuffle && crossesWarp) {
        unsupported(op, "forced warp shuffle cannot exchange values across warps");
    }
    MovementWork work;
    if (!crossesWarp &&
        (forceShuffle || cvtNeedsWarpShuffle(src, dst))) {
        work = warpWork(src, dst, bitwidth);
    } else {
        work = sharedWork(src, dst, bitwidth);
    }

    CostVector cost = builder.zeroVector();
    // Instruction-equivalent proxies: these are not exposed wait-cycle costs.
    addWork(builder, cost, CostType::INT32,
            "triton_gpu.layout.shuffle_cost", work.shuffles);
    addWork(builder, cost, CostType::INT32,
            "triton_gpu.layout.select_cost", work.selects);
    addWork(builder, cost, CostType::L1,
            "triton_gpu.layout.shared_byte_cost", work.sharedBytes);
    addWork(builder, cost, CostType::INT32,
            "triton_gpu.layout.warp_sync_cost", work.warpSyncs);
    addWork(builder, cost, CostType::INT32,
            "triton_gpu.layout.cta_barrier_cost", work.ctaBarriers);
    return cost;
}
