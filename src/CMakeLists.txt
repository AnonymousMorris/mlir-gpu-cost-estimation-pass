cmake_minimum_required(VERSION 3.20)
project(Skeleton LANGUAGES CXX C)

# LLVM uses C++17.
set(CMAKE_CXX_STANDARD 17)

set(TRITON_LLVM_DIR "/home/morris/.triton/llvm/llvm-ubuntu-x64" CACHE PATH "Path to Triton's LLVM/MLIR install")
set(LLVM_DIR "${TRITON_LLVM_DIR}/lib/cmake/llvm" CACHE PATH "Path to LLVMConfig.cmake" FORCE)
set(MLIR_DIR "${TRITON_LLVM_DIR}/lib/cmake/mlir" CACHE PATH "Path to MLIRConfig.cmake" FORCE)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

set(TRITON_SOURCE_DIR "/home/morris/code/triton" CACHE PATH "Path to Triton source checkout")
set(TRITON_BUILD_DIR "/home/morris/code/triton/build/cmake.linux-x86_64-cpython-3.14" CACHE PATH "Path to Triton CMake build directory")

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

# Include the part of LLVM's CMake libraries that defines
# `add_llvm_pass_plugin`.
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Use LLVM's preprocessor definitions, include directories, and library search
# paths.
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_llvm_library(
    MyPass
    MODULE
    pass.cpp
    core.cpp
    triton.cpp
    CostIRBuilder.cpp
    Loop.cpp

    OUTPUT_NAME "libMyPass"
)

target_include_directories(MyPass PRIVATE
    ${TRITON_SOURCE_DIR}/include
    ${TRITON_BUILD_DIR}/include
)
