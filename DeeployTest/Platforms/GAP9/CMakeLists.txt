# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})



set(${SDKCONFIG_FILE} ${CMAKE_CURRENT_LIST_DIR}/sdk.config)

file(GLOB_RECURSE SOURCES
    src/CycleCounter.c
    src/deeploytest.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
# add_executable(${ProjectId} ${SOURCES})
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc)

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)
add_gvsoc_emulation(${ProjectId} "gap9.evk")

# RW: Waive sign comparison warnings from pulp_nn_utils.h
target_compile_options(network PRIVATE
    -Wno-sign-compare
    -Wno-pointer-sign
    -Wno-unknown-pragmas
    -Wno-error
  )

target_link_options(${ProjectId} PRIVATE
    -Wl,--print-memory-usage
)
link_compile_dump(${TESTNAME})
