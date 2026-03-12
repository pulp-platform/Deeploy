# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# XDNA2 (AIE2p) kernel library
#
# Compiles AIE C++ kernels using the llvm-aie (Peano) cross-compiler.
# Exports a CMake target `xdna2_kernels` that other targets can depend on,
# and sets XDNA2_KERNEL_OBJECTS in the parent scope.
# ---------------------------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# --- Resolve llvm-aie (Peano) install dir ---
set(LLVM_AIE_INSTALL_DIR "$ENV{LLVM_AIE_INSTALL_DIR}" CACHE PATH "llvm-aie (Peano) install dir")
if(NOT LLVM_AIE_INSTALL_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import aie.utils.config; print(aie.utils.config.peano_install_dir());"
        OUTPUT_VARIABLE LLVM_AIE_INSTALL_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
if(NOT LLVM_AIE_INSTALL_DIR)
    message(FATAL_ERROR "[XDNA2] Could not find llvm-aie install dir. "
        "Please set the environment variable LLVM_AIE_INSTALL_DIR or install the llvm-aie wheel.")
endif()

# --- Resolve mlir-aie include dir (aie_api headers) ---
if(NOT MLIR_AIE_INCLUDE_DIR)
    if(DEFINED ENV{MLIR_AIE_INCLUDE_DIR})
        set(MLIR_AIE_INCLUDE_DIR $ENV{MLIR_AIE_INCLUDE_DIR})
    else()
        execute_process(
            COMMAND ${Python3_EXECUTABLE}
                    -c "import aie.utils.config; print(aie.utils.config.cxx_header_path());"
            OUTPUT_VARIABLE MLIR_AIE_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _aie_cfg_result
        )
        if(NOT _aie_cfg_result EQUAL 0 OR NOT MLIR_AIE_INCLUDE_DIR)
            message(FATAL_ERROR "[XDNA2] Could not query aie.utils.config.cxx_header_path(). "
                "Please set the environment variable MLIR_AIE_INCLUDE_DIR or install the mlir-aie wheel.")
        endif()
    endif()
endif()

set(LLVM_AIE_CLANG "${LLVM_AIE_INSTALL_DIR}/bin/clang++")

message(STATUS "[XDNA2 Kernels] LLVM_AIE_INSTALL_DIR = ${LLVM_AIE_INSTALL_DIR}")
message(STATUS "[XDNA2 Kernels] MLIR_AIE_INCLUDE_DIR = ${MLIR_AIE_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Compile AIE kernels
# ---------------------------------------------------------------------------
file(GLOB XDNA2_KERNEL_SOURCES "${CMAKE_CURRENT_LIST_DIR}/kernels/*.cc")

set(XDNA2_KERNEL_OBJECTS "")

foreach(KERNEL_SRC ${XDNA2_KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
    set(KERNEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.o")

    add_custom_command(
        OUTPUT  "${KERNEL_OBJ}"
        COMMAND "${LLVM_AIE_CLANG}"
                --target=aie2p-none-unknown-elf
                "-I${MLIR_AIE_INCLUDE_DIR}"
                -std=c++20
                -Wno-parentheses
                -Wno-attributes
                -Wno-macro-redefined
                -Wno-empty-body
                -O2
                -DNDEBUG
                -c "${KERNEL_SRC}"
                -o "${KERNEL_OBJ}"
        DEPENDS "${KERNEL_SRC}"
        COMMENT "[XDNA2] Compiling AIE kernel: ${KERNEL_NAME}.cc -> ${KERNEL_NAME}.o"
        VERBATIM
    )

    list(APPEND XDNA2_KERNEL_OBJECTS "${KERNEL_OBJ}")
endforeach()

add_custom_target(xdna2_kernels DEPENDS ${XDNA2_KERNEL_OBJECTS})

# Export kernel objects to parent scope so the testbench CMake can use them
set(XDNA2_KERNEL_OBJECTS "${XDNA2_KERNEL_OBJECTS}" PARENT_SCOPE)
