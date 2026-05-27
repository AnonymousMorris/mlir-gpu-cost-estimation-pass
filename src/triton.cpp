#include "CostConfig.h"
#include "CostIRBuilder.h"
#include "gpuSpec.h"
#include "triton.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <algorithm>
#include <cassert>
#include <optional>

using namespace mlir;

int64_t elements_per_thread(Value value) {
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

Value scale_cost(CostIRBuilder &costBuilder, llvm::StringRef name,
                 int64_t scale) {
    Value cost = costBuilder.addCostArgument(name);
    if (scale == 1) {
        return cost;
    }
    return costBuilder.mul(cost, costBuilder.constantCost(scale));
}

llvm::StringRef tensor_cost_name(Operation &op) {
    auto costIt = NamedTensorOpCost.find(op.getName().getStringRef());
    assert(costIt != NamedTensorOpCost.end() && "missing tensor op cost name");
    return std::get<llvm::StringRef>(costIt->second);
}

int64_t tensor_k_dim(Value value) {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType || tensorType.getRank() == 0) {
        return 1;
    }
    return tensorType.getShape().back();
}

Value analyze_triton_load(CostIRBuilder &costBuilder, triton::LoadOp loadOp,
                          const GpuSpec &) {
    return scale_cost(costBuilder, tensor_cost_name(*loadOp.getOperation()),
                      elements_per_thread(loadOp.getResult()));
}

Value analyze_triton_store(CostIRBuilder &costBuilder, triton::StoreOp storeOp,
                           const GpuSpec &) {
    return scale_cost(costBuilder, tensor_cost_name(*storeOp.getOperation()),
                      elements_per_thread(storeOp.getValue()));
}

Value analyze_triton_dot(CostIRBuilder &costBuilder, triton::DotOp dotOp,
                         const GpuSpec &) {
    int64_t outputElemsPerThread = elements_per_thread(dotOp.getD());
    int64_t k = tensor_k_dim(dotOp.getA());
    return scale_cost(costBuilder, tensor_cost_name(*dotOp.getOperation()),
                      outputElemsPerThread * k);
}

Value analyze_triton_addptr(CostIRBuilder &costBuilder, triton::AddPtrOp addPtrOp,
                            const GpuSpec &) {
    return scale_cost(costBuilder, tensor_cost_name(*addPtrOp.getOperation()),
                      elements_per_thread(addPtrOp.getOffset()));
}

Value analyze_triton_broadcast(CostIRBuilder &costBuilder,
                               triton::BroadcastOp broadcastOp,
                               const GpuSpec &) {
    return scale_cost(costBuilder, tensor_cost_name(*broadcastOp.getOperation()),
                      elements_per_thread(broadcastOp.getResult()));
}

Value analyze_triton_expand_dims(CostIRBuilder &costBuilder,
                                 triton::ExpandDimsOp expandDimsOp,
                                 const GpuSpec &) {
    return scale_cost(costBuilder, tensor_cost_name(*expandDimsOp.getOperation()),
                      elements_per_thread(expandDimsOp.getSrc()));
}

Value analyze_triton_splat(CostIRBuilder &costBuilder, triton::SplatOp splatOp,
                           const GpuSpec &) {
    return scale_cost(costBuilder, tensor_cost_name(*splatOp.getOperation()),
                      elements_per_thread(splatOp.getResult()));
}

Value analyze_triton_make_range(CostIRBuilder &costBuilder,
                                triton::MakeRangeOp makeRangeOp,
                                const GpuSpec &) {
    return scale_cost(costBuilder, tensor_cost_name(*makeRangeOp.getOperation()),
                      elements_per_thread(makeRangeOp.getResult()));
}

Value analyze_ttg_local_alloc(CostIRBuilder &costBuilder,
                              triton::gpu::LocalAllocOp localAllocOp,
                              const GpuSpec &) {
    if (Value src = localAllocOp.getSrc()) {
        return scale_cost(costBuilder,
                          tensor_cost_name(*localAllocOp.getOperation()),
                          elements_per_thread(src));
    }
    return scale_cost(costBuilder, tensor_cost_name(*localAllocOp.getOperation()),
                      1);
}

Value analyze_ttg_local_load(CostIRBuilder &costBuilder,
                             triton::gpu::LocalLoadOp localLoadOp,
                             const GpuSpec &) {
    return scale_cost(costBuilder, tensor_cost_name(*localLoadOp.getOperation()),
                      elements_per_thread(localLoadOp.getResult()));
}

Value analyze_ttg_convert_layout(CostIRBuilder &costBuilder,
                                 triton::gpu::ConvertLayoutOp convertLayoutOp,
                                 const GpuSpec &) {
    int64_t elemsPerThread =
        std::max<int64_t>(elements_per_thread(convertLayoutOp.getSrc()),
                          elements_per_thread(convertLayoutOp.getResult()));
    return scale_cost(costBuilder,
                      tensor_cost_name(*convertLayoutOp.getOperation()),
                      elemsPerThread);
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
