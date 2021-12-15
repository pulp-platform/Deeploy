set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin/riscv32-unknown-elf)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}-ar)
set(SIZE ${TOOLCHAIN_PREFIX}-size)


if(xpulpimg)
    add_compile_options(
        -march=rv32imaXpulpimg
        -Wa,-march=rv32imaXpulpimg
    )
    add_link_options(
        -march=rv32imaXpulpimg
        -Wa,-march=rv32imaXpulpimg
    )
else()
    add_compile_options(
        -march=rv32ima
        -Wa,-march=rv32ima
    )
    add_link_options(
        -march=rv32ima
        -Wa,-march=rv32ima
    )
endif()

add_compile_options(
    -mabi=ilp32
    -mcmodel=medany
    -mtune=mempool

    # -falign-loops=32
    # -falign-jumps=32
    # Turn of optimization that lead to known problems
    -fno-tree-loop-distribute-patterns
    -fno-builtin-memcpy
    -fno-builtin-memset

    -fno-builtin-printf
    -fno-common

    -static
)

add_link_options(
    -mabi=ilp32
    -mcmodel=medany
    -mtune=mempool

    # Turn of optimization that lead to known problems
    -fno-tree-loop-distribute-patterns
    -fno-builtin-memcpy
    -fno-builtin-memset

    -fno-builtin-printf
    -fno-common

    -static
    -nostartfiles
)

link_libraries(
  -lm
  -lgcc
)


add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_GCC__)
