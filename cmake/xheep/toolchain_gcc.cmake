# Copyright (C) 2026 EPFL.
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# File: toolchain_gcc.cmake
# Author: Mohammad Hossein Nikkhah
# Description: 

set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin/riscv32-unknown-elf)

set(CMAKE_SYSTEM_NAME Generic)

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}-ar)
set(SIZE ${TOOLCHAIN_PREFIX}-size)

set(ISA rv32imc_zicsr CACHE STRING "X-HEEP RISC-V ISA")
set(CMAKE_SYSTEM_PROCESSOR ${ISA} CACHE STRING "X-HEEP RISC-V ISA")


set(CMAKE_EXECUTABLE_SUFFIX ".elf")

add_compile_options(
  -march=${ISA}
  -ffunction-sections
  -fdata-sections
  -O2
  -g
  -MMD
  -MP
)

add_link_options(
  -MMD
  -MP
  -march=${ISA}
  -nostartfiles
  -nostdlib
  -Wl,--print-memory-usage
)

link_libraries(
  -lc
  -lm
  -lgcc
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_GCC__)
