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
    {"arith.constant", 0.0},
    {"scf.yield", 0.0},
    {"tt.get_program_id", 0.0},
    {"tt.return", 0.0},
    {"vector.shape_cast", 0.0},
};

inline const CostConfig NamedOpCosts = {
    {"arith.addf", "arith.addf"},
    {"arith.addi", "arith.addi"},
    {"arith.andi", "arith.andi"},
    {"arith.cmpi", "arith.cmpi"},
    {"arith.divsi", "arith.divsi"},
    {"arith.divui", "arith.divui"},
    {"arith.fma", "arith.fma"},
    {"arith.minsi", "arith.minsi"},
    {"arith.mulf", "arith.mulf"},
    {"arith.muli", "arith.muli"},
    {"arith.remsi", "arith.remsi"},
    {"arith.remui", "arith.remui"},
    {"arith.subi", "arith.subi"},
    {"arith.truncf", "arith.truncf"},
    {"arith.uitofp", "arith.uitofp"},
    {"index.castu", "index.castu"},
    {"vector.extract", "vector.extract"},
    {"vector.extract_strided_slice", "vector.extract_strided_slice"},
};

inline const CostConfig MemOpCost = {
};

inline const CostConfig NamedTensorOpCost = {
    // Global/shared memory load instructions, plus address/predicate operands.
    {"tt.load", "triton.load_cost"},
    // Global/shared memory store instructions, plus address/predicate operands.
    {"tt.store", "triton.store_cost"},
    // Matrix multiply-accumulate instructions, e.g. NVIDIA mma/wgmma.
    {"tt.dot", "triton.dot_cost"},
    // Pointer/address integer arithmetic; often folded into address generation.
    {"tt.addptr", "triton.addptr_cost"},
    // Shape-only broadcast in IR; typically no standalone instruction.
    {"tt.broadcast", 0.0},
    // Shape-only rank change in IR; typically no standalone instruction.
    {"tt.expand_dims", 0.0},
    // Uniform scalar reuse or register moves, depending on lowering.
    {"tt.splat", "triton.splat_cost"},
    // Per-lane index materialization, usually constants/register arithmetic.
    {"tt.make_range", "triton.make_range_cost"},
    // Shared/local memory allocation; usually no instruction, but consumes storage.
    {"ttg.local_alloc", "triton_gpu.local_alloc_cost"},
    // Local/shared memory load or register extraction, depending on layout.
    {"ttg.local_load", "triton_gpu.local_load_cost"},
    // Layout movement via register shuffles and/or shared-memory traffic.
    {"ttg.convert_layout", "triton_gpu.convert_layout_cost"},
};
