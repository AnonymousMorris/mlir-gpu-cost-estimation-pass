cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0219)
    cmake_policy(SET CMP0219 NEW)
endif()

project(Skeleton LANGUAGES CXX C)

include(CTest)

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
    Reduction.cpp
    CostIRBuilder.cpp
    Loop.cpp

    OUTPUT_NAME "libMyPass"
)

target_include_directories(MyPass PRIVATE
    ${TRITON_SOURCE_DIR}/include
    ${TRITON_BUILD_DIR}/include
)

if(BUILD_TESTING)
    find_program(TRITON_OPT_EXECUTABLE NAMES triton-opt)
    if(TRITON_OPT_EXECUTABLE)
        add_test(
            NAME loop-count-division-bound
            COMMAND
                ${TRITON_OPT_EXECUTABLE}
                --load-pass-plugin=$<TARGET_FILE:MyPass>
                "--pass-pipeline=builtin.module(my-cost-analysis{func-name=division_loop_count})"
                -o ${CMAKE_CURRENT_BINARY_DIR}/division_loop_count.out.mlir
                ${CMAKE_CURRENT_SOURCE_DIR}/../tests/triton/division_loop_count.ttgir
        )
        set_tests_properties(loop-count-division-bound PROPERTIES
            PASS_REGULAR_EXPRESSION
                "arith.divsi %[0-9]+, %c64_i32 : i32"
        )

        add_test(
            NAME symbolic-cost-argument-roles
            COMMAND
                ${TRITON_OPT_EXECUTABLE}
                --load-pass-plugin=$<TARGET_FILE:MyPass>
                "--pass-pipeline=builtin.module(my-cost-analysis{func-name=division_loop_count})"
                -o ${CMAKE_CURRENT_BINARY_DIR}/symbolic_cost_argument_roles.out.mlir
                ${CMAKE_CURRENT_SOURCE_DIR}/../tests/triton/division_loop_count.ttgir
        )
        set_tests_properties(symbolic-cost-argument-roles PROPERTIES
            PASS_REGULAR_EXPRESSION
                "cost.kind = \"weight\".*cost.kind = \"runtime\""
        )

        add_test(
            NAME shared-memory-l1-cost-category
            COMMAND
                ${TRITON_OPT_EXECUTABLE}
                --load-pass-plugin=$<TARGET_FILE:MyPass>
                "--pass-pipeline=builtin.module(my-cost-analysis{func-name=tiled_matmul_kernel})"
                -o ${CMAKE_CURRENT_BINARY_DIR}/shared_memory_l1.out.mlir
                ${CMAKE_CURRENT_SOURCE_DIR}/../tests/triton/tiled_matmul.ttgir
        )
        set_tests_properties(shared-memory-l1-cost-category PROPERTIES
            PASS_REGULAR_EXPRESSION "cost.name = \"l1\""
        )

        find_program(UV_EXECUTABLE NAMES uv)
        if(UV_EXECUTABLE)
            add_test(
                NAME shared-memory-l1-cost-routing
                COMMAND
                    ${UV_EXECUTABLE}
                    run
                    --project ${CMAKE_CURRENT_SOURCE_DIR}/../analysis
                    python
                    ${CMAKE_CURRENT_SOURCE_DIR}/../analysis/tests/verify_l1_pass.py
                    ${TRITON_OPT_EXECUTABLE}
                    $<TARGET_FILE:MyPass>
                    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/triton/tiled_matmul.ttgir
            )

            add_test(
                NAME global-memory-element-widths
                COMMAND
                    ${UV_EXECUTABLE}
                    run
                    --project ${CMAKE_CURRENT_SOURCE_DIR}/../analysis
                    python
                    ${CMAKE_CURRENT_SOURCE_DIR}/../analysis/tests/verify_memory_widths_pass.py
                    ${TRITON_OPT_EXECUTABLE}
                    $<TARGET_FILE:MyPass>
                    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/triton/memory_element_widths.ttgir
            )

            add_test(
                NAME shared-memory-element-widths
                COMMAND
                    ${UV_EXECUTABLE}
                    run
                    --project ${CMAKE_CURRENT_SOURCE_DIR}/../analysis
                    python
                    ${CMAKE_CURRENT_SOURCE_DIR}/../analysis/tests/verify_l1_widths_pass.py
                    ${TRITON_OPT_EXECUTABLE}
                    $<TARGET_FILE:MyPass>
                    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/triton/shared_element_widths.ttgir
            )

            add_test(
                NAME triton-reduction-cost
                COMMAND
                    ${UV_EXECUTABLE}
                    run
                    --project ${CMAKE_CURRENT_SOURCE_DIR}/../analysis
                    python
                    ${CMAKE_CURRENT_SOURCE_DIR}/../analysis/tests/verify_reduction_pass.py
                    ${TRITON_OPT_EXECUTABLE}
                    $<TARGET_FILE:MyPass>
                    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/triton/reduction.ttgir
            )
        endif()
    endif()
endif()
