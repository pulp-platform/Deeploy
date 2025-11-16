# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

add_compile_definitions(
  DEEPLOY_PULP_PLATFORM
)

set(DEEPLOY_ARCH PULP)

add_compile_options(
  -ffast-math
)

add_link_options(
  -ffast-math
  -Wl,--gc-sections
)
