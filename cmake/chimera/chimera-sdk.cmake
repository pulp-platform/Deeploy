# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

add_compile_definitions(
  DEEPLOY_CHIMERA_PLATFORM
)

set(DEEPLOY_ARCH CHIMERA)

add_compile_options(
  -ffast-math
)

add_link_options(
  -ffast-math
  -Wl,--gc-sections
)

set(ABI ilp32)
set(ISA_CLUSTER_SNITCH rv32im)
set(ISA_HOST rv32imc)

