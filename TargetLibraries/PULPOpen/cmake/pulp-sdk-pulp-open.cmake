include(cmake/pulp-sdk-base.cmake)

set(PULP_SDK_HOME $ENV{PULP_SDK_HOME})

set(PULP_OPEN_COMPILE_FLAGS
  -DCONFIG_PULP
  -DCONFIG_BOARD_VERSION_PULP
  -DCONFIG_PROFILE_PULP
  -DUSE_HYPERFLASH
  -DUSE_HYPERRAM
  -DPULP_CHIP_STR=pulp
)

set(PULP_OPEN_INCLUDES
  ${PULP_SDK_HOME}/rtos/pulpos/pulp/include/pos/chips/pulp
)

set(PULP_SDK_PULP_OPEN_C_SOURCE
  ${PULP_SDK_HOME}/rtos/pulpos/common/kernel/fll-v1.c
  ${PULP_SDK_HOME}/rtos/pulpos/common/kernel/freq-domains.c
  ${PULP_SDK_HOME}/rtos/pulpos/pulp/kernel/chips/pulp/soc.c
)

set_source_files_properties(${PULP_SDK_PULP_OPEN_ASM_SOURCE} PROPERTIES COMPILE_FLAGS -DLANGUAGE_ASSEMBLY)
add_library(pulp-sdk OBJECT ${PULP_SDK_BASE_C_SOURCE} ${PULP_SDK_BASE_ASM_SOURCE} ${PULP_SDK_PULP_OPEN_C_SOURCE} ${PULP_SDK_PULP_OPEN_ASM_SOURCE})

set(PULP_SDK_COMPILE_FLAGS ${PULP_OPEN_COMPILE_FLAGS} ${PULP_SDK_BASE_COMPILE_FLAGS})
set(PULP_SDK_INCLUDES ${PULP_OPEN_INCLUDES} ${PULP_SDK_BASE_INCLUDE})

target_include_directories(pulp-sdk SYSTEM PUBLIC ${PULP_SDK_INCLUDES} ${PULP_OPEN_INCLUDES})
target_compile_options(pulp-sdk PUBLIC ${PULP_SDK_COMPILE_FLAGS})
target_compile_options(pulp-sdk PRIVATE
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
target_compile_options(pulp-sdk INTERFACE
  -Wno-unused-function
)


set(PULP_OPEN_LINK_OPTIONS
  -Wl,--gc-sections
  -L${PULP_SDK_HOME}/rtos/pulpos/pulp/kernel
  -Tchips/pulp/link.ld
)

target_link_libraries(pulp-sdk PUBLIC
  ${PULP_OPEN_LINK_OPTIONS}
)
