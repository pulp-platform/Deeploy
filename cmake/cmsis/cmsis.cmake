# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

add_compile_definitions(
    DEEPLOY_CMSIS_PLATFORM
)

set(DEEPLOY_ARCH CMSIS)

add_compile_options(
  -ffast-math
)

add_link_options(
  -ffast-math
  -Wl,--gc-sections
)
