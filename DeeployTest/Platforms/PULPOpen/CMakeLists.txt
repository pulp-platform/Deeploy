# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

set(PULP_DMA MCHAN CACHE STRING "Cluster DMA engine (MCHAN, IDMA)")
set_property(CACHE PULP_DMA PROPERTY STRINGS MCHAN IDMA)
if(NOT PULP_DMA MATCHES "^(MCHAN|IDMA)$")
  message(FATAL_ERROR "[Deeploy PULP-OPEN] Unknown PULP_DMA '${PULP_DMA}'. Supported: MCHAN, IDMA")
endif()

# Generated code and build must agree on the cluster DMA
file(STRINGS ${GENERATED_SOURCE}/Network.h _dmaDefine REGEX "#define DEEPLOY_PULP_DMA_")
if(NOT _dmaDefine MATCHES "DEEPLOY_PULP_DMA_${PULP_DMA}$")
  message(FATAL_ERROR "[Deeploy PULP-OPEN] PULP_DMA=${PULP_DMA} but Network.h has '${_dmaDefine}'. "
                      "Regenerate with --dma or reconfigure with -DPULP_DMA.")
endif()
message(STATUS "[Deeploy PULP-OPEN] Cluster DMA = ${PULP_DMA}")

file(GLOB_RECURSE SOURCES
    src/CycleCounter.c
    src/deeploytest.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc)

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
add_gvsoc_emulation(${ProjectId} "pulp-open")

link_compile_dump(${TESTNAME})
