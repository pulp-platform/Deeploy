set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/llvm-objcopy)
set(CMAKE_OBJDUMP  ${TOOLCHAIN_PREFIX}/llvm-objdump)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}/llvm-ar)
set(SIZE ${TOOLCHAIN_PREFIX}/llvm-size)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(xpulpimg)
  message(FATAL_ERROR "Xpulpimg extension is not supported for this compiler!")
else()
    add_compile_options(
        -march=rv32ima
    )
    add_link_options(
        -march=rv32ima
    )
endif()

add_compile_options(
  --target=riscv32-unknown-elf
  --sysroot=${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv

  -mabi=ilp32
  -mcmodel=medany
  -mcpu=mempool-rv32
  -mllvm
  -misched-topdown

  -std=gnu99

  -fno-builtin-memcpy
  -fno-builtin-memset

  -ffast-math
  -fno-builtin-printf
  -fno-common
  -fdiagnostics-color=always

  -Wunused-variable
  -Wconversion
  -Wall
  -Wextra

  -static
)

add_link_options(
  --target=riscv32-unknown-elf
  --sysroot=${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv

  -mabi=ilp32
  -mcmodel=medany
  -mcpu=mempool-rv32
  -std=gnu99

  -fno-builtin-memcpy
  -fno-builtin-memset

  -ffast-math
  -fno-builtin-printf
  -fno-common
  -fdiagnostics-color=always

  -Wunused-variable
  -Wconversion
  -Wall
  -Wextra

  -static
  -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32im/
)

link_libraries(
  -lm
)

add_compile_definitions(__LINK_LD)
add_compile_definitions(__TOOLCHAIN_LLVM__)