set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)

set(LLVM_TAG llvm)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objdump)

set(ISA cortex-m4)
set(PE 8)
set(FC 1)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

add_compile_options(
  -target armv7m-none-eabi
  -mcpu=${ISA}
  -ffunction-sections
  -fdata-sections
  -fomit-frame-pointer
  -fno-exceptions
  -fno-rtti
  -mno-relax
  -O2
  -g3
  -DNUM_CORES=${NUM_CORES}
  -MMD
  -MP
  -I${TOOLCHAIN_INSTALL_DIR}/picolibc/arm/include
)

add_link_options(
  -target armv7m-none-eabi
  -MMD
  -MP
  -mcpu=${ISA}
  -L${TOOLCHAIN_INSTALL_DIR}/picolibc/arm/lib
  -Tpicolibc.ld
  -v
  -Wl,--defsym=__flash=0x00000000
  -Wl,--defsym=__flash_size=0x400000
  -Wl,--defsym=__ram=0x20000000
  -Wl,--defsym=__ram_size=0x400000
  -Wl,--defsym=__stack_size=0x4000
  #-z norelro
)

link_libraries(
  -lm
  -lc
  -lcrt0-semihost
  -lsemihost
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_LLVM__)
