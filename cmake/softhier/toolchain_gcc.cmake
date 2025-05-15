add_compile_definitions(
    DEEPLOY_SOFTHIER_PLATFORM
)

set(SOFTHIER_TOOLCHAIN_INSTALL_DIR $ENV{SOFTHIER_INSTALL_DIR}/third_party/toolchain/install)
set(TOOLCHAIN_PREFIX ${SOFTHIER_TOOLCHAIN_INSTALL_DIR}/bin/riscv32-unknown-elf)

# Building for bare metal system
set(CMAKE_SYSTEM_NAME Generic)

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}-ar)
set(SIZE ${TOOLCHAIN_PREFIX}-size)

set(ISA rv32imafdv_zfh)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

add_compile_options(
  -mabi=ilp32d 
  -mcmodel=medlow
  -march=${ISA}
  -g 
  -O3 
  -ffast-math
  -fno-builtin 
  -fno-tree-vectorize 
  -fno-common 
  -ffunction-sections
  -fno-strict-aliasing
)

add_link_options(
  -march=${ISA}
  -nostartfiles
  -Wl,--gc-sections 
)

link_libraries(
  -lm
  -lgcc
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_GCC__)
