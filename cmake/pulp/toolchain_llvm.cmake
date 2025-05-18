set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)

set(LLVM_TAG llvm)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objdump)

set(ISA rv32imf_xpulpv2)
set(PE 8)
set(FC 1)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

add_compile_options(
  -target riscv32-unknown-elf
  -march=${ISA}
  -ffunction-sections
  -fdata-sections
  -fomit-frame-pointer
  -mno-relax
  -O3
  -DNUM_CORES=${NUM_CORES}
  -MP
  --sysroot=${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/rv32imc
  -fno-builtin-memcpy
  -fno-builtin-memset
)

add_link_options(
  -target riscv32-unknown-elf
  -MP
  -nostartfiles
  -march=${ISA}
  -L${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv/rv32imc/lib
  -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32imc/
  -z norelro
  -fno-builtin-memcpy
  -fno-builtin-memset
)

link_libraries(
  -lm
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_LLVM__)
