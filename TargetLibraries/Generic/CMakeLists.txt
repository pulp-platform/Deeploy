# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
    "src/**"
)

add_deeploy_library(deeploybasic STATIC ${SOURCES})

target_include_directories(deeploybasic
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/inc
)
