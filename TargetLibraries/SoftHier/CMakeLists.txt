# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
    "src/**"
)

include(cmake/softhier-runtime.cmake)
add_deeploy_library(deeploysofthier STATIC ${SOURCES})

target_include_directories(deeploysofthier
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/inc
)

target_compile_options(deeploysofthier PRIVATE
  -DDEEPLOY_SOFTHIER_PLATFORM
  -Wno-implicit-function-declaration
  -Wno-implicit-int-conversion
  -Wno-sign-conversion
  -Wno-sign-compare
)

target_link_libraries(deeploysofthier INTERFACE softhier-sdk)
target_sources(deeploysofthier INTERFACE $<TARGET_OBJECTS:softhier-sdk>)