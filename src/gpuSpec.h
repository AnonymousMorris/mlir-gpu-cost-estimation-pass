#pragma once

#include <cstdint>
#include <string>

struct GpuSpec {
    std::string backend;
    std::string targetArch;
    int64_t numCTAs;
    int64_t numWarps;
    int64_t threadsPerWarp;

    bool isAmpere() const {
        return backend == "cuda" &&
               (targetArch == "80" || targetArch == "86" || targetArch == "87");
    }

    int64_t threadsPerCTA() const {
        return numWarps * threadsPerWarp;
    }
};
