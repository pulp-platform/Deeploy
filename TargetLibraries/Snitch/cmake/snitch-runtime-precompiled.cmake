set(SNITCH_RUNTIME_BASE_INCLUDE
  ${SNITCH_RUNTIME_HOME}/src
  ${SNITCH_RUNTIME_HOME}/api
  ${SNITCH_RUNTIME_HOME}/../deps/riscv-opcodes # TODO: generate riscv-opcodes whatever
  ${SNITCH_RUNTIME_HOME}/../math/include # TODO: generate riscv-opcodes whatever
  ${SNITCH_HOME}/sw/math/arch/riscv64/
)

set(SNITCH_RUNTIME_OMP_INCLUDE
  ${SNITCH_RUNTIME_HOME}/src/omp
  ${SNITCH_RUNTIME_HOME}/api/omp
)

if(banshee_simulation)
  set(SNITCH_CLUSTER_INCLUDE
      ${SNITCH_CLUSTER_HOME}/sw/runtime/common
      ${SNITCH_CLUSTER_HOME}/sw/runtime/banshee
      ${SNITCH_CLUSTER_HOME}/sw/runtime/banshee/src
	)
  set(SNITCH_CLUSTER_LINK_INCLUDE
    ${SNITCH_CLUSTER_HOME}/sw/math/build
    ${SNITCH_CLUSTER_HOME}/sw/runtime/banshee
    ${SNITCH_CLUSTER_HOME}/sw/runtime/banshee/build
  )
else()
  set(SNITCH_CLUSTER_INCLUDE
    ${SNITCH_CLUSTER_HOME}/sw/runtime/common
    ${SNITCH_CLUSTER_HOME}/sw/runtime/rtl
    ${SNITCH_CLUSTER_HOME}/sw/runtime/rtl/src
  )
  set(SNITCH_CLUSTER_LINK_INCLUDE
    ${SNITCH_CLUSTER_HOME}/sw/math/build
    ${SNITCH_CLUSTER_HOME}/sw/runtime/rtl
    ${SNITCH_CLUSTER_HOME}/sw/runtime/rtl/build
  )
endif()

set(SNITCH_CLUSTER_LINK_OPTIONS
  -Wl,--gc-sections
  -T ${SNITCH_RUNTIME_HOME}/base.ld
)

set(SNITCH_RUNTIME_INCLUDE ${SNITCH_RUNTIME_BASE_INCLUDE} ${SNITCH_RUNTIME_OMP_INCLUDE} ${SNITCH_CLUSTER_INCLUDE})


add_library(snitch-runtime INTERFACE)
target_link_directories(snitch-runtime INTERFACE ${SNITCH_CLUSTER_LINK_INCLUDE})
target_link_libraries(snitch-runtime INTERFACE ${SNITCH_CLUSTER_LINK_OPTIONS} libsnRuntime.a libmath.a)