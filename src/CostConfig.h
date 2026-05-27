#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <variant>

using CostSpec = std::variant<double, llvm::StringRef>;
using CostConfig = llvm::StringMap<CostSpec>;

// This is where the costs for operations are set
// We allow either setting a constant cost or leaving
// the cost parameterized in which case the final cost
// would be expressed in terms of the parameterized cost
// name.
// {opname, F64 cost} or {opname, cost_name}
inline const CostConfig SimpleOpCosts = {
    // {"xegpu.create_nd_tdesc", 0.0},
    // {"arith.addf", 1.0},
    // {"arith.addi", 1.0},
    // {"arith.cmpi", 1.0},
    // {"arith.divui", 1.0},
    // {"arith.fma", 1.0},
    // {"arith.muli", 1.0},
    // {"arith.remui", 1.0},
    // {"arith.truncf", 1.0},
    // {"arith.uitofp", 1.0},
    // {"index.castu", 1.0},
    // {"vector.extract", 1.0},
    // {"vector.extract_strided_slice", 1.0},
    // {"vector.shape_cast", 1.0},
};

inline const CostConfig NamedOpCosts = {
    {"arith.addf", "arith.addf"},
    {"arith.addi", "arith.addi"},
    {"arith.cmpi", "arith.cmpi"},
    {"arith.divui", "arith.divui"},
    {"arith.fma", "arith.fma"},
    {"arith.muli", "arith.muli"},
    {"arith.remui", "arith.remui"},
    {"arith.truncf", "arith.truncf"},
    {"arith.uitofp", "arith.uitofp"},
    {"index.castu", "index.castu"},
    {"vector.extract", "vector.extract"},
    {"vector.extract_strided_slice", "vector.extract_strided_slice"},
    {"vector.shape_cast", "vector.shape_cast"},
};

inline const CostConfig MemOpCost = {
    {"xegpu.dpas", "dpas_cost"},
};

inline const CostConfig NamedTensorOpCost = {
    {"tt.load", "triton.load_cost"},
    {"tt.store", "triton.store_cost"},
    {"tt.dot", "triton.dot_cost"},
    {"tt.addptr", "triton.addptr_cost"},
    {"tt.broadcast", "triton.broadcast_cost"},
    {"tt.expand_dims", "triton.expand_dims_cost"},
    {"tt.splat", "triton.splat_cost"},
    {"tt.make_range", "triton.make_range_cost"},
    {"ttg.local_alloc", "triton_gpu.local_alloc_cost"},
    {"ttg.local_load", "triton_gpu.local_load_cost"},
    {"ttg.convert_layout", "triton_gpu.convert_layout_cost"},
};
