# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/llvm-objcopy)
set(CMAKE_OBJDUMP  ${TOOLCHAIN_PREFIX}/llvm-objdump)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}/llvm-ar)
set(SIZE ${TOOLCHAIN_PREFIX}/llvm-size)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(xpulpimg)
  message(FATAL_ERROR "Xpulpimg extension is not supported for this compiler!")
else()
    add_compile_options(
        -march=rv32ima
    )
    add_link_options(
        -march=rv32ima
    )
endif()

add_compile_options(
  --target=riscv32-unknown-elf

  -mabi=ilp32
  -mcmodel=medany
  -mcpu=mempool-rv32
  -mllvm
  -misched-topdown

  -std=gnu99

  -fno-builtin-memcpy
  -fno-builtin-memset

  -ffast-math
  -fno-builtin-printf
  -fno-common
  -fdiagnostics-color=always

  -Wunused-variable
  -Wconversion
  -Wall
  -Wextra

  -static
  -isystem ${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/rv32ima/include
)

add_link_options(
  --target=riscv32-unknown-elf

  -mabi=ilp32
  -mcmodel=medany
  -mcpu=mempool-rv32
  -std=gnu99

  -fno-builtin-memcpy
  -fno-builtin-memset

  -ffast-math
  -fno-builtin-printf
  -fno-common
  -fdiagnostics-color=always

  -Wunused-variable
  -Wconversion
  -Wall
  -Wextra

  -static
  -L${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/rv32ima/lib
  -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32ima/
)

link_libraries(
  -lm
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_LLVM__)