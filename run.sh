#!/usr/bin/bash

mlir-opt --load-pass-plugin=build/libMyPass.so \
    --pass-pipeline='builtin.module(my-cost-analysis)' \
    -o /dev/null \
    tests/xegpu/gemm.mlir
