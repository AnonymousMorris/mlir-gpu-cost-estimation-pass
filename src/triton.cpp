#include "CostIRBuilder.h"
#include "gpuSpec.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <optional>

using namespace mlir;

int64_t elements_per_thread(Value &value) {
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
        return triton::gpu::getTotalElemsPerThread(tensorType);
    }
    return 1;
}

// Return the max element per thread for all the operands
int64_t elements_per_thread(Operation &op) {
    int64_t maxElemsPerThread = 1;

    for (Value operand : op.getOperands()) {
        maxElemsPerThread =
            std::max<int64_t>(maxElemsPerThread, elements_per_thread(operand));
    }
    for (Value result : op.getResults()) {
        maxElemsPerThread =
            std::max<int64_t>(maxElemsPerThread, elements_per_thread(result));
    }

    return maxElemsPerThread;
}

namespace {

Value placeholder_cost(CostIRBuilder &costBuilder) {
    return costBuilder.constantCost(1.0);
}

Value analyze_triton_load(CostIRBuilder &costBuilder, triton::LoadOp loadOp,
                          const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_triton_store(CostIRBuilder &costBuilder, triton::StoreOp storeOp,
                           const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_triton_dot(CostIRBuilder &costBuilder, triton::DotOp dotOp,
                         const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_triton_addptr(CostIRBuilder &costBuilder, triton::AddPtrOp addPtrOp,
                            const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_triton_broadcast(CostIRBuilder &costBuilder,
                               triton::BroadcastOp broadcastOp,
                               const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_triton_expand_dims(CostIRBuilder &costBuilder,
                                 triton::ExpandDimsOp expandDimsOp,
                                 const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_triton_splat(CostIRBuilder &costBuilder, triton::SplatOp splatOp,
                           const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_triton_make_range(CostIRBuilder &costBuilder,
                                triton::MakeRangeOp makeRangeOp,
                                const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_ttg_local_alloc(CostIRBuilder &costBuilder,
                              triton::gpu::LocalAllocOp localAllocOp,
                              const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_ttg_local_load(CostIRBuilder &costBuilder,
                             triton::gpu::LocalLoadOp localLoadOp,
                             const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

Value analyze_ttg_convert_layout(CostIRBuilder &costBuilder,
                                 triton::gpu::ConvertLayoutOp convertLayoutOp,
                                 const GpuSpec &gpu) {
    return placeholder_cost(costBuilder);
}

} // namespace


std::optional<Value> analyze_triton_tensor_op(CostIRBuilder &costBuilder,
                                              Operation &op,
                                              const GpuSpec &gpu) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        return analyze_triton_load(costBuilder, loadOp, gpu);
    }

    if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
        return analyze_triton_store(costBuilder, storeOp, gpu);
    }

    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
        return analyze_triton_dot(costBuilder, dotOp, gpu);
    }

    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return analyze_triton_addptr(costBuilder, addPtrOp, gpu);
    }

    if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(op)) {
        return analyze_triton_broadcast(costBuilder, broadcastOp, gpu);
    }

    if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(op)) {
        return analyze_triton_expand_dims(costBuilder, expandDimsOp, gpu);
    }

    if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
        return analyze_triton_splat(costBuilder, splatOp, gpu);
    }

    if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
        return analyze_triton_make_range(costBuilder, makeRangeOp, gpu);
    }

    if (auto localAllocOp = dyn_cast<triton::gpu::LocalAllocOp>(op)) {
        return analyze_ttg_local_alloc(costBuilder, localAllocOp, gpu);
    }

    if (auto localLoadOp = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        return analyze_ttg_local_load(costBuilder, localLoadOp, gpu);
    }

    if (auto convertLayoutOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
        return analyze_ttg_convert_layout(costBuilder, convertLayoutOp, gpu);
    }

    return {};
}
