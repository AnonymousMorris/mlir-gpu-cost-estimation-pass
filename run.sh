#!/usr/bin/bash

triton-opt --load-pass-plugin=build/libMyPass.so \
    --pass-pipeline='builtin.module(my-cost-analysis{func-name=tiled_matmul_kernel})' \
    -o /dev/null \
    tests/triton/tiled_matmul.ttgir
