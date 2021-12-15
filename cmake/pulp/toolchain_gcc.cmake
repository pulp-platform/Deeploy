set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin/riscv32-unknown-elf)

set(CMAKE_SYSTEM_NAME Generic)

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}-ar)
set(SIZE ${TOOLCHAIN_PREFIX}-size)

set(ISA rv32imc_zfinx_xpulpv3)
set(PE 8)
set(FC 1)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

add_compile_options(
  -march=${ISA}
  -ffunction-sections
  -fdata-sections
  -fomit-frame-pointer
  -fno-jump-tables
  -fno-tree-loop-distribute-patterns
  -O3
  -DNUM_CORES=${NUM_CORES}
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
  -lm
  -lgcc
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_GCC__)
