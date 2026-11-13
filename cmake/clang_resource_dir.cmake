# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Sets CLANG_RESOURCE_DIR to the resource directory of CMAKE_C_COMPILER
# (e.g. <llvm>/lib/clang/18), where the Makefile installs compiler-rt.
# Asking the compiler keeps toolchain files independent of the LLVM version.
#
# CMake also re-reads toolchain files for its internal try_compile projects,
# where TOOLCHAIN_INSTALL_DIR is not set; the query is skipped there.

if(EXISTS ${CMAKE_C_COMPILER})
  execute_process(
    COMMAND ${CMAKE_C_COMPILER} -print-resource-dir
    OUTPUT_VARIABLE CLANG_RESOURCE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _clang_resource_dir_result
  )
  if(NOT _clang_resource_dir_result EQUAL 0 OR NOT CLANG_RESOURCE_DIR)
    message(FATAL_ERROR "Could not query the resource directory of ${CMAKE_C_COMPILER}")
  endif()
endif()
