# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)

set(LLVM_TAG llvm)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/riscv32-unknown-elf-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objdump)

set(ISA rv32imc CACHE STRING "X-HEEP RISC-V ISA")


set(ABI ilp32 CACHE STRING "X-HEEP RISC-V ABI")

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

set(CMAKE_AR "${TOOLCHAIN_PREFIX}/llvm-ar")
set(CMAKE_RANLIB "${TOOLCHAIN_PREFIX}/llvm-ranlib")


add_compile_options(
  -target riscv32-unknown-elf
  -march=${ISA}
  -mabi=${ABI}
  -ffunction-sections
  -fdata-sections
  -fomit-frame-pointer
  -mno-relax
  -O3
  -MP
  --sysroot=${TOOLCHAIN_INSTALL_DIR}/riscv32-unknown-elf
  -fno-builtin-memcpy
  -fno-builtin-memset
)

add_link_options(
  -fuse-ld=lld
  -target riscv32-unknown-elf
  -MP
  -nostartfiles
  -march=${ISA}
  -mabi=${ABI}
  --sysroot=${TOOLCHAIN_INSTALL_DIR}/riscv32-unknown-elf
  -L${TOOLCHAIN_INSTALL_DIR}/riscv32-unknown-elf/lib
  -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/19/include
  -z norelro
  -fno-builtin-memcpy
  -fno-builtin-memset
)

link_libraries(
  -lm
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_LLVM__)
