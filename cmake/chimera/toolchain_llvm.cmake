set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)

set(LLVM_TAG llvm)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objdump)
set(CMAKE_LINKER ${TOOLCHAIN_PREFIX}/ld.lld)
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

# Enable WHOLE_ARCHIVE feature
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE 
"-Wl,--whole-archive <LIBRARY> -Wl,--no-whole-archive"
)
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED True)

set(CROSS_COMPILE_HOST "riscv32-unknown-elf")

set(ISA rv32imc)

add_compile_options(
    -target riscv32-unknown-elf
    -MP
    -march=${ISA}
    -mno-relax
    -menable-experimental-extensions
    -mabi=ilp32d
    -mcmodel=medany
    -fno-builtin-printf
    -fno-builtin-sqrtf
    -fno-common
    -fopenmp
    -ftls-model=local-exec
    -DNUM_CORES=${NUM_CORES}
    -isystem ${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/${ISA}/include
)

add_link_options(
    -target riscv32-unknown-elf
    -MP
    -fuse-ld=lld
    -nostartfiles
    -nostdlib
    -L${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/${ISA}/lib
    -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/${ISA}
)

link_libraries(
    -lc
    -lclang_rt.builtins-riscv32
)

# Required by math library to avoid conflict with stdint definition
add_compile_definitions(__LINK_LLD)
add_compile_definitions(__TOOLCHAIN_LLVM__)
add_compile_definitions(__DEFINED_uint64_t)
