include(cmake/snitch-runtime-base.cmake)

set(SNITCH_CLUSTER_HOME ${SNITCH_HOME}/target/snitch_cluster/sw/)

set(SNITCH_CLUSTER_COMPILE_FLAGS
)


if(banshee_simulation)
	set(SNITCH_CLUSTER_INCLUDE
		${SNITCH_CLUSTER_HOME}/sw/runtime/common
    ${SNITCH_CLUSTER_HOME}/sw/runtime/banshee
    ${SNITCH_CLUSTER_HOME}/sw/runtime/banshee/src
	)
  set(SNITCH_CLUSTER_C_SOURCE
    ${SNITCH_CLUSTER_HOME}/sw/runtime/banshee/src/snitch_cluster_start.S
    ${SNITCH_CLUSTER_HOME}/sw/runtime/banshee/src/snrt.c
  )
else()
  set(SNITCH_CLUSTER_INCLUDE
    ${SNITCH_CLUSTER_HOME}/sw/runtime/common
    ${SNITCH_CLUSTER_HOME}/sw/runtime/rtl
    ${SNITCH_CLUSTER_HOME}/sw/runtime/rtl/src
  )
  set(SNITCH_CLUSTER_C_SOURCE
    ${SNITCH_CLUSTER_HOME}/sw/runtime/rtl/src/snitch_cluster_start.S
    ${SNITCH_CLUSTER_HOME}/sw/runtime/rtl/src/snrt.c
  )
endif()


set(SNITCH_CLUSTER_LINK_OPTIONS
  -Wl,--gc-sections
  -T ${SNITCH_RUNTIME_HOME}/base.ld
)


add_library(snitch-runtime STATIC ${SNITCH_RUNTIME_BASE_C_SOURCE} ${SNITCH_RUNTIME_BASE_ASM_SOURCE} ${SNITCH_CLUSTER_C_SOURCE})

set(SNITCH_RUNTIME_COMPILE_FLAGS ${SNITCH_RUNTIME_BASE_COMPILE_FLAGS} ${SNITCH_CLUSTER_COMPILE_FLAGS})
set(SNITCH_RUNTIME_INCLUDE ${SNITCH_RUNTIME_BASE_INCLUDE} ${SNITCH_RUNTIME_OMP_INCLUDE} ${SNITCH_CLUSTER_INCLUDE})

target_include_directories(snitch-runtime SYSTEM PUBLIC ${SNITCH_RUNTIME_INCLUDE})

target_compile_options(snitch-runtime PUBLIC ${SNITCH_RUNTIME_COMPILE_FLAGS})
target_compile_options(snitch-runtime PRIVATE
  -O2
  -Wno-sign-conversion
  -Wno-unused-function
  -Wno-unused-parameter
  -Wno-conversion
  -Wno-sign-conversion
  -Wno-unused-variable
  -Wno-sign-compare
  -Wno-return-type
  -fno-inline-functions
)
target_compile_options(snitch-runtime INTERFACE
  -Wno-unused-function
)

target_link_libraries(snitch-runtime PUBLIC
  ${SNITCH_CLUSTER_LINK_OPTIONS}
)