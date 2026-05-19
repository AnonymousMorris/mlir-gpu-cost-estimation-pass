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
    {"arith.muli", 1.0},
};
inline const CostConfig NamedOpCosts = {
    {"xegpu.dpas", "dpas_cost"},
};
