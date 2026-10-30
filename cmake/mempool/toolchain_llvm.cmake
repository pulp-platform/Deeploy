# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/llvm-objcopy)
set(CMAKE_OBJDUMP  ${TOOLCHAIN_PREFIX}/llvm-objdump)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}/clang)
include(${CMAKE_CURRENT_LIST_DIR}/../clang_resource_dir.cmake)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}/llvm-ar)
set(SIZE ${TOOLCHAIN_PREFIX}/llvm-size)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(xpulpimg)
  message(FATAL_ERROR "Xpulpimg extension is not supported for this compiler!")
else()
    add_compile_options(
        -march=rv32ima_xmempool
    )
    add_link_options(
        -march=rv32ima_xmempool
    )
endif()

add_compile_options(
  --target=riscv32-unknown-elf

  -mabi=ilp32
  -mcmodel=medany
  -mcpu=mempool-rv32
  # LLVM 19 replaced -misched-topdown with -misched-prera-direction=topdown
  "SHELL:-mllvm $<IF:$<VERSION_LESS:$<C_COMPILER_VERSION>,19>,-misched-topdown,-misched-prera-direction=topdown>"
  # Banshee corrupts the trailing bytes of an ELF load segment whose size is not
  # a multiple of 4. GlobalMerge (on for RISC-V in recent LLVM) reorders the
  # static .l2 constants, which can leave an odd-sized int8 tensor at the end of
  # the last segment (e.g. Kernels/Integer/Conv/DW_2D_RQ). Keep source order.
  "SHELL:-mllvm -enable-global-merge=false"

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
  # startup code comes from the MemPool runtime; clang >= 20 would add crt0.o
  -nostartfiles
  -nodefaultlibs
  -L${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/rv32ima/lib
  -L${CLANG_RESOURCE_DIR}/lib/baremetal/rv32ima/
)

# clang >= 19 links compiler-rt by an absolute per-target path that ignores -L,
# so link it explicitly to select the ISA-specific build.
link_libraries(
  -lm
  -lc
  -lclang_rt.builtins-riscv32
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_LLVM__)