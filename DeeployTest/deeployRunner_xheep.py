#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import sys

from testUtils.deeployRunner import main


def setup_xheep_defaults(parser):
    """
    Setting up default values for X-HEEP platform
    """
    parser.set_defaults(
        toolchain="GCC",
        toolchain_install_dir="/app/tools/riscv/",
        skipsim=False,       ## TODO : Remove this default after adding verilator simulation
        cmake=[
            "-DXHEEP_HOME=/app/x-heep",
            "-DXHEEP_TARGET=sim",
            # "-DXHEEP_LINKER=flash_load",
            "-DXHEEP_LINKER=on_chip",
        ],
    )


if __name__ == "__main__":
    #TODO : make the co-simulation with X-HEEP repository (Verilator)
    sys.exit(main(default_platform = "Xheep", default_simulator = "verilator", tiling_enabled = False, parser_setup_callback=setup_xheep_defaults))
