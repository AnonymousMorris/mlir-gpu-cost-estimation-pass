cmake_minimum_required(VERSION 3.20)
project(Skeleton LANGUAGES CXX C)

# LLVM uses C++17.
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

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
    CostIRBuilder.cpp
    Loop.cpp

    OUTPUT_NAME "libMyPass"
)
