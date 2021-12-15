set(MEMPOOL_HOME $ENV{MEMPOOL_HOME})
set(MEMPOOL_RUNTIME_HOME ${MEMPOOL_HOME}/software/runtime)

set(MEMPOOL_RUNTIME_C_SOURCE
  ${MEMPOOL_RUNTIME_HOME}/alloc.c
  ${MEMPOOL_RUNTIME_HOME}/dma.c
  ${MEMPOOL_RUNTIME_HOME}/printf.c
  ${MEMPOOL_RUNTIME_HOME}/serial.c
  ${MEMPOOL_RUNTIME_HOME}/string.c
  ${MEMPOOL_RUNTIME_HOME}/synchronization.c
  )

set(MEMPOOL_RUNTIME_ASM_SOURCE
  ${MEMPOOL_RUNTIME_HOME}/crt0.S
)

set(MEMPOOL_RUNTIME_INCLUDE
  ${MEMPOOL_RUNTIME_HOME}
  ${MEMPOOL_RUNTIME_HOME}/target/${mempool_flavour}
)

set(MEMPOOL_RUNTIME_COMPILE_FLAGS
  -D__riscv__
  -D__builtin_shuffle=__builtin_pulp_shuffle2h
)

if (${MEMPOOL_USE_OMP})
set(MEMPOOL_RUNTIME_OMP_C_SOURCE
  ${MEMPOOL_RUNTIME_HOME}/omp/barrier.c
  ${MEMPOOL_RUNTIME_HOME}/omp/critical.c
  ${MEMPOOL_RUNTIME_HOME}/omp/loop.c
  ${MEMPOOL_RUNTIME_HOME}/omp/parallel.c
  ${MEMPOOL_RUNTIME_HOME}/omp/sections.c
  ${MEMPOOL_RUNTIME_HOME}/omp/single.c
  ${MEMPOOL_RUNTIME_HOME}/omp/work.c
)

set(MEMPOOL_RUNTIME_OMP_INCLUDE
  ${MEMPOOL_RUNTIME_HOME}/omp/
)
endif()

if (${MEMPOOL_USE_HALIDE})
set(MEMPOOL_RUNTIME_HALIDE_C_SOURCE
  ${MEMPOOL_RUNTIME_HOME}/halide/halide_runtime.c
)

set(MEMPOOL_RUNTIME_HALIDE_INCLUDE
  ${MEMPOOL_RUNTIME_HOME}/halide/
)
endif()

get_directory_property(DirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/arch.ld
  MAIN_DEPENDENCY ${MEMPOOL_RUNTIME_HOME}/target/${mempool_flavour}/arch.ld.c
  COMMAND ${CMAKE_C_COMPILER}
  -P -E "$<$<BOOL:${DirDefs}>:-D$<JOIN:${DirDefs},;-D>>"
  ${MEMPOOL_RUNTIME_HOME}/target/${mempool_flavour}/arch.ld.c
  -o ${CMAKE_BINARY_DIR}/arch.ld
  COMMAND_EXPAND_LISTS
  COMMENT "Generate arch.ld from arch.ld.c (${CMAKE_BINARY_DIR})"
  VERBATIM)

# Create a consumer target for the linker script.
# So previous `add_custom_command` will have an effect.
add_custom_target(linkerscript DEPENDS ${CMAKE_BINARY_DIR}/arch.ld)

set_source_files_properties(${MEMPOOL_RUNTIME_ASM_SOURCE} PROPERTIES COMPILE_FLAGS -DLANGUAGE_ASSEMBLY)
add_library(mempool-runtime OBJECT
  ${MEMPOOL_RUNTIME_C_SOURCE}
  ${MEMPOOL_RUNTIME_ASM_SOURCE}
  ${MEMPOOL_RUNTIME_OMP_C_SOURCE}
  ${MEMPOOL_RUNTIME_HALIDE_C_SOURCE}
)

target_include_directories(mempool-runtime SYSTEM PUBLIC
  ${MEMPOOL_RUNTIME_INCLUDE}
  ${MEMPOOL_RUNTIME_OMP_INCLUDE}
  ${MEMPOOL_RUNTIME_HALIDE_INCLUDE}
)
target_compile_options(mempool-runtime PUBLIC ${MEMPOOL_RUNTIME_COMPILE_FLAGS})
target_compile_options(mempool-runtime PRIVATE
  -O2
  -fno-inline
  -fno-common
)
target_compile_options(mempool-runtime INTERFACE
  -Wno-unused-function
)

set(MEMPOOL_LINK_OPTIONS
  -Wl,--gc-sections
  -L${CMAKE_BINARY_DIR}
  -T${MEMPOOL_RUNTIME_HOME}/target/${mempool_flavour}/link.ld
)

target_link_libraries(mempool-runtime PUBLIC
  ${MEMPOOL_LINK_OPTIONS}
)

# Make executable to depend on that target.
# So, the check whether to relink the executable will be performed
# after possible rebuilding the linker script.
add_dependencies(mempool-runtime linkerscript)