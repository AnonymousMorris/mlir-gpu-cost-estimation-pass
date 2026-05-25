#pragma once

#include <cstdint>
#include <string>

struct GpuSpec {
    std::string backend;
    std::string targetArch;
    int64_t numCTAs;
    int64_t numWarps;
    int64_t threadsPerWarp;

    int64_t threadsPerCTA() const {
        return numWarps * threadsPerWarp;
    }
};
