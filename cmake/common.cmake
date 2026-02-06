# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(use_dma 1 CACHE STRING "Enable DMA trasfers")

add_compile_definitions(
    USE_DMA=${use_dma}
)

add_compile_options(
    -std=gnu99

    -fdiagnostics-color=always

    -Wunused-variable
    -Wconversion
    -Wall
    -Wextra

    -O2
    -g
    -ffunction-sections
    -fdata-sections
)

add_link_options(
    -std=gnu99

    -fdiagnostics-color=always

    -Wunused-variable
    -Wconversion
    -Wall
    -Wextra
)
