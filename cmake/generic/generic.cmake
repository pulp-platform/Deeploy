# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

add_compile_definitions(
    DEEPLOY_GENERIC_PLATFORM
)

if(APPLE)
  add_link_options(
    -Wl,-dead_strip
  )
else()
  add_link_options(
    -Wl,--gc-sections
  )
endif()
