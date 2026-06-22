
set(CMAKE_SYSTEM_NAME Generic)

# Crucial: Point CMake to the specialized Clang toolchain instead of system cc
set(SPATZ_TOOLCHAIN_DIR ${SPATZ_HOME}/sw/toolchain/llvm-project/build/bin)

set(CMAKE_C_COMPILER   ${SPATZ_TOOLCHAIN_DIR}/clang)
set(CMAKE_CXX_COMPILER ${SPATZ_TOOLCHAIN_DIR}/clang++)
set(CMAKE_ASM_COMPILER ${SPATZ_TOOLCHAIN_DIR}/clang)
set(CMAKE_OBJCOPY ${SPATZ_TOOLCHAIN_DIR}/llvm-objcopy)
set(CMAKE_OBJDUMP ${SPATZ_TOOLCHAIN_DIR}/llvm-objdump)
set(CMAKE_LINKER ${SPATZ_TOOLCHAIN_DIR}/ld.lld)
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

set(ISA rv32imafdvzfh_xdma)

# Compile options based on user's manual compilation commands
add_compile_options(
    -target riscv32-unknown-elf
    # -MP
    -mcpu=snitch
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
    # Newlib headers: prefer $GCC_INSTALL_DIR (set by util/iis-env.sh to the
    # cluster's spatz-gcc) over a source-built GNU toolchain inside spatz.
    # -isystem $ENV{GCC_INSTALL_DIR}/riscv32-unknown-elf/include
    -isystem ${SPATZ_HOME}/sw/toolchain/riscv-gnu-toolchain/riscv-newlib/newlib/libc/include
    
    # Optimization and debug
    -O3 
    -g
)

# Link options matching user command
add_link_options(
    # -target riscv32-unknown-elf
    -mcpu=snitch
    -march=${ISA}
    -mabi=ilp32d
    -mcmodel=small
    
    -fuse-ld=lld
    -nostartfiles

    -ffast-math
    -fno-common
    -fno-builtin-printf
     
    -static 
    -Wl,-z,norelro 
    -Wl,--gc-sections 
    -Wl,--no-relax

    --gcc-toolchain=/usr/pack/riscv-1.0-kgf/spatz-gcc-7.1.1
)

# libsnRuntime-cluster.a is handled by our target_link_libraries(deeployspatz INTERFACE spatz-runtime)
link_libraries(
    -lm -lgcc -lm -lgcc
)
