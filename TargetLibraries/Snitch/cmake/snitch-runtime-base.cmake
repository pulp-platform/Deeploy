

option(SNRT_USE_OMP "Use OpenMP" OFF)

set(SNITCH_HOME $ENV{SNITCH_HOME})

set(SNITCH_RUNTIME_HOME ${SNITCH_HOME}/sw/snRuntime)

set(SNITCH_RUNTIME_BASE_C_SOURCE
  ${SNITCH_RUNTIME_HOME}/src/alloc.c
  ${SNITCH_RUNTIME_HOME}/src/cls.c
  ${SNITCH_RUNTIME_HOME}/src/cluster_interrupts.c
  ${SNITCH_RUNTIME_HOME}/src/dm.c
  ${SNITCH_RUNTIME_HOME}/src/dma.c
  ${SNITCH_RUNTIME_HOME}/src/global_interrupts.c
  ${SNITCH_RUNTIME_HOME}/src/printf.c
  ${SNITCH_RUNTIME_HOME}/src/start.c
  ${SNITCH_RUNTIME_HOME}/src/sync.c
  ${SNITCH_RUNTIME_HOME}/src/team.c
)

set(SNITCH_RUNTIME_BASE_ASM_SOURCE

)

set(SNITCH_RUNTIME_BASE_INCLUDE
  ${SNITCH_RUNTIME_HOME}/src
  ${SNITCH_RUNTIME_HOME}/api
  ${SNITCH_RUNTIME_HOME}/../deps/riscv-opcodes # TODO: generate riscv-opcodes whatever
  ${SNITCH_RUNTIME_HOME}/../math/include # TODO: generate riscv-opcodes whatever
)

set(SNITCH_RUNTIME_BASE_COMPILE_FLAGS
  -D__riscv__
)

if (${SNRT_USE_OMP})

set(SNITCH_RUNTIME_OMP_C_SOURCE
  ${SNITCH_RUNTIME_HOME}/src/omp/eu.c
  ${SNITCH_RUNTIME_HOME}/src/omp/kmp.c
  ${SNITCH_RUNTIME_HOME}/src/omp/omp.c
)

set(SNITCH_RUNTIME_OMP_INCLUDE
${SNITCH_RUNTIME_HOME}/src/omp
${SNITCH_RUNTIME_HOME}/api/omp
)
endif()

set_source_files_properties(${SNITCH_RUNTIME_BASE_ASM_SOURCE} PROPERTIES COMPILE_FLAGS -DLANGUAGE_ASSEMBLY)