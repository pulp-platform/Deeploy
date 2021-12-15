set(TOOLCHAIN_PREFIX arm-none-eabi)

set(CMAKE_SYSTEM_NAME Generic)

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}-ar)
set(SIZE ${TOOLCHAIN_PREFIX}-size)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

add_compile_options(
  -mthumb
  -ffunction-sections
  -fdata-sections
  -fomit-frame-pointer
  -MMD
  -MP
  -std=c99
  -Wall
  -g
  -O2
  -I${TOOLCHAIN_INSTALL_DIR}/picolibc/arm/include
)

add_link_options(
  -mthumb
  -nostartfiles
  -static
  -MMD
  -MP
  -Wl,--print-memory-usage
  -L${TOOLCHAIN_INSTALL_DIR}/picolibc/arm/lib
  -Tpicolibc.ld
  -Wl,--defsym=__flash=0x00000000
  -Wl,--defsym=__flash_size=0x400000
  -Wl,--defsym=__ram=0x20000000
  -Wl,--defsym=__ram_size=0x400000
  -Wl,--defsym=__stack_size=0x4000
)

link_libraries(
  -lm
  -lc
  -lcrt0-semihost
  -lsemihost
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_GCC__)
