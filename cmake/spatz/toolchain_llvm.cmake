# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)

set(LLVM_TAG llvm)

# Crucial: Point CMake to the specialized Clang toolchain instead of system cc
set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objdump)
set(CMAKE_LINKER ${TOOLCHAIN_PREFIX}/ld.lld)
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

# ISA definition from user command
set(ISA rv32imafdvzfh_xdma_xfquarter)

# Compile options based on user's manual compilation commands
add_compile_options(
    -target riscv32-unknown-elf
    -MP
    -mcpu=snitch
    # -mcmodel=small  # User used small, Snitch uses medany. Keeping user's choice can be risky if code is large.
    # Safe compromise: use medany unless 'small' is strictly required, but user command had small.
    # deeploy typically uses medany. Let's stick to user's flag if explicit, or Snitch defaults.
    # User command: -mcmodel=small
    -mcmodel=small
    
    -ffast-math
    -fno-builtin-printf
    -fno-common
    -falign-loops=16 
    -ffunction-sections
    -Wextra
    
    # LLVM specific flags from user command
    -mllvm -misched-topdown 
    -menable-experimental-extensions
    -mno-relax
    
    -march=${ISA}
    -mabi=ilp32d
    -isystem ${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/rv32imafd/include
    
    # Optimization and debug
    -O3 
    -g
    
    # Include paths will be handled by CMake target_include_directories
    # But we need riscv-opcodes if not standard
    # -I.../riscv-opcodes is usually handled by Snitch/Spatz runtime includes
)

# Link options matching user command
add_link_options(
    -target riscv32-unknown-elf
    -MP
    -mcpu=snitch
    -march=${ISA}
    -mabi=ilp32d
    -mcmodel=small
    
    -fuse-ld=lld
    -nostartfiles
    -nostdlib
    -ffast-math
    -fno-common
    -fno-builtin-printf
    
    -Wl,-z,norelro 
    -Wl,--gc-sections 
    -Wl,--no-relax
    -L${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/rv32imafd/lib
    -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32imafd
    
    # User had explicit gcc toolchain path: --gcc-toolchain=/usr/pack/... 
    # In Docker/Deeploy we typically use packaged libs or environment variables.
    # We will try to rely on the container's environment first.
)

# User command linked: -lm -lgcc -lm -lgcc libsnRuntime-cluster.a
# libsnRuntime is handled by our target_link_libraries(deeployspatz ... snitch-runtime)
link_libraries(
    -lc
    -lclang_rt.builtins-riscv32
)

# Required by math library to avoid conflict with stdint definition
add_definitions(-D__DEFINED_intptr_t)
