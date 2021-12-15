set(TARGET_CPU "cortex-m4" CACHE STRING "Target CPU")
set(CPU cortex-m4)
set(FPU fpv4-sp-d16)
set(FABI soft)

add_compile_options(
  -mcpu=${CPU}
  -mfpu=${FPU}
  -mfloat-abi=${FABI}
)

add_link_options(
  -mcpu=${CPU}
  -mfpu=${FPU}
  -mfloat-abi=${FABI}
)

macro(add_binary_dump name)
  add_custom_target(bin_${name}
    DEPENDS ${name}
    COMMAND ${CMAKE_OBJCOPY} -Obinary ${CMAKE_BINARY_DIR}/bin/${name} ${CMAKE_BINARY_DIR}/bin/${name}.bin
    COMMENT "Dumping raw binary"
    POST_BUILD
    USES_TERMINAL
    VERBATIM
  )
endmacro()

macro(add_qemu_emulation name)
  add_custom_target(qemu_${name}
    DEPENDS bin_${name}
    COMMAND qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -monitor null -semihosting --semihosting-config enable=on,target=native -kernel ${CMAKE_BINARY_DIR}/bin/${name}.bin -serial stdio -nographic
    COMMENT "Simulating deeploytest with QEMU"
    POST_BUILD
    USES_TERMINAL
    VERBATIM
  )
endmacro()
