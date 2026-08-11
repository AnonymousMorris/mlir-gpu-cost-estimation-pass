#include "Reduction.h"

#include "gpuSpec.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <cstdint>

using namespace mlir;

CostVector analyze_region(CostIRBuilder &costBuilder, Region &region,
                          const GpuSpec &gpu);

namespace {

constexpr llvm::StringLiteral WarpShuffleCostName =
    "triton.reduce_shuffle_cost";

struct ReductionLayoutInfo {
    // Number of independent reduction outputs held in registers by each
    // thread after its local reduction is complete.
    int64_t accumulatorsPerThread;

    // Number of unique values along the reduction axis held by one thread.
    int64_t threadElements;

    // Number of unique reduction-axis values distributed across lanes in one
    // warp. Its base-two logarithm is the number of shuffle/combine stages.
    int64_t warpElements;

    // Number of unique reduction-axis values still distributed across warps or
    // CTAs after the first warp reduction.
    int64_t interWarpElements;
};

int64_t axis_elements(const triton::LinearLayout &layout, StringAttr inputDim,
                      unsigned axis) {
    const auto &bases = layout.getBases().lookup(inputDim);
    int64_t axisBases = llvm::count_if(bases, [axis](const auto &basis) {
        return basis[axis] != 0;
    });
    assert(axisBases < 63 && "reduction layout exceeds int64_t range");
    return int64_t{1} << axisBases;
}

ReductionLayoutInfo get_reduction_layout_info(triton::ReduceOp reduceOp) {
    auto srcType = cast<RankedTensorType>(reduceOp.getSrcs().front().getType());
    unsigned axis = reduceOp.getAxis();
    MLIRContext *ctx = reduceOp.getContext();
    auto kRegister = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");

    // This mirrors the first step of Triton's reduction lowering. Registers
    // containing duplicate broadcast values do not take part in the reduction.
    triton::LinearLayout layout = triton::gpu::toLinearLayout(srcType);
    layout = triton::actionRemoveBroadcastedRegs(layout).apply(layout);

    auto linearEncoding = triton::gpu::LinearEncodingAttr::get(ctx, layout);
    auto registerElementsPerDim =
        linearEncoding.basesPerDim(kRegister, /*skipBroadcast=*/true);
    int64_t threadElements = registerElementsPerDim[axis];
    assert(threadElements > 0 && "expected reduction elements in registers");

    int64_t registersPerThread = layout.getInDimSize(kRegister);
    assert(registersPerThread % threadElements == 0 &&
           "reduction axis must divide the register values");

    return {
        registersPerThread / threadElements,
        threadElements,
        axis_elements(layout, kLane, axis),
        axis_elements(layout, kWarp, axis) *
            axis_elements(layout, kBlock, axis),
    };
}

CostVector scale_cost(CostIRBuilder &costBuilder, CostVector cost,
                      int64_t scale) {
    if (scale == 0) {
        return costBuilder.zeroVector();
    }
    if (scale == 1) {
        return cost;
    }
    return costBuilder.mul(cost, costBuilder.constantCost(scale));
}

CostVector analyze_thread_reduction(CostIRBuilder &costBuilder,
                                    CostVector combinerCost,
                                    const ReductionLayoutInfo &layoutInfo) {
    // Reducing N thread-local values requires N - 1 applications of the
    // combiner for each independent accumulator.
    int64_t combines = layoutInfo.accumulatorsPerThread *
                       (layoutInfo.threadElements - 1);
    return scale_cost(costBuilder, combinerCost, combines);
}

CostVector analyze_warp_reduction(CostIRBuilder &costBuilder,
                                  triton::ReduceOp reduceOp,
                                  CostVector combinerCost,
                                  const ReductionLayoutInfo &layoutInfo) {
    // Model Triton's general shuffle-tree path: each participating lane basis
    // adds one shuffle and one combiner stage. Target-specific packed or redux
    // instructions can refine this symbolic model later.
    int64_t warpStages = llvm::Log2_64(layoutInfo.warpElements);
    int64_t combines = layoutInfo.accumulatorsPerThread * warpStages;
    CostVector cost = scale_cost(costBuilder, combinerCost, combines);

    // Every source/result component is shuffled once at each stage. Keep the
    // communication cost symbolic because its instruction cost is target- and
    // type-dependent.
    int64_t shuffles = combines * reduceOp.getNumOperands();
    if (shuffles != 0) {
        Value shuffleCost = costBuilder.addCostArgument(WarpShuffleCostName);
        shuffleCost = costBuilder.mul(shuffleCost,
                                      costBuilder.constantCost(shuffles));
        cost = costBuilder.add(
            cost, costBuilder.costVector(CostType::FP32, shuffleCost));
    }
    return cost;
}

CostVector analyze_inter_warp_and_layout_conversion(
    CostIRBuilder &costBuilder, const ReductionLayoutInfo &layoutInfo) {
    // TODO: Model the convert-layout/shared-memory rounds used to move warp and
    // block bases into lanes, followed by their additional warp reductions.
    // layoutInfo.interWarpElements already records how much axis data remains.
    (void)layoutInfo;
    return costBuilder.zeroVector();
}

} // namespace

CostVector analyze_triton_reduce(CostIRBuilder &costBuilder,
                                 triton::ReduceOp reduceOp,
                                 const GpuSpec &gpu) {
    ReductionLayoutInfo layoutInfo = get_reduction_layout_info(reduceOp);

    // The region represents one application of the user-provided combiner. It
    // is shared by all hierarchy levels and can contain multiple operations or
    // produce multiple reduction results.
    CostVector combinerCost =
        analyze_region(costBuilder, reduceOp.getCombineOp(), gpu);
    CostVector threadCost =
        analyze_thread_reduction(costBuilder, combinerCost, layoutInfo);
    CostVector warpCost = analyze_warp_reduction(
        costBuilder, reduceOp, combinerCost, layoutInfo);
    CostVector interWarpCost = analyze_inter_warp_and_layout_conversion(
        costBuilder, layoutInfo);

    return costBuilder.sumCosts({threadCost, warpCost, interWarpCost});
}
