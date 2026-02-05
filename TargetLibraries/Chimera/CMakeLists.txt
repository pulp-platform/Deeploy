# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
  "src/**"
)

# Import chimera-open target
add_subdirectory(
  $ENV{CHIMERA_SDK_HOME}
  ${CMAKE_BINARY_DIR}/chimera-open
)

add_deeploy_library(deeploychimera STATIC ${SOURCES})
target_include_directories(deeploychimera
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
)

target_link_libraries(deeploychimera INTERFACE runtime_host hal_host)