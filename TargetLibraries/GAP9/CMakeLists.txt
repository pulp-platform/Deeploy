# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
  "src/**"
)

if(NOT DEFINED ENV{GAP_SDK_HOME})
  message(FATAL_ERROR "Environment variable GAP_SDK_HOME not set.")
endif()

add_deeploy_library(deeploygap9 STATIC ${SOURCES})
target_include_directories(deeploygap9
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
)

target_compile_options(deeploygap9 PUBLIC
    -DNUM_CORES=${NUM_CORES}
  )

target_compile_options(deeploygap9 PRIVATE
  -Wno-implicit-function-declaration
  -Wno-implicit-int-conversion
  -Wno-sign-conversion
  -Wno-sign-compare
)

target_link_libraries(deeploygap9 PUBLIC pmsis)
