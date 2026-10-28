#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys

from testUtils.deeployRunner import main


class CMakeCacheVariableAction(argparse.Action):
    """Translate Python command-line arguments into CMake cache variables."""

    def __init__(self, option_strings, dest, cmake_variable, **kwargs):
        self.cmake_variable = cmake_variable
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string = None):
        prefix = f"-D{self.cmake_variable}="

        # Copy the existing CMake arguments and replace this setting, if present.
        cmake_args = list(getattr(namespace, "cmake", None) or [])
        cmake_args = [arg for arg in cmake_args if not arg.startswith(prefix)]
        cmake_args.append(f"{prefix}{value}")

        setattr(namespace, "cmake", cmake_args)
        setattr(namespace, self.dest, value)


def setup_xheep_defaults(parser):
    """Configure X-HEEP-specific defaults and command-line arguments.

    Sets the default X-HEEP target to simulation and the default program
    memory layout to on-chip memory. It also registers options for selecting
    a different X-HEEP target or memory layout.

    Args:
        parser (argparse.ArgumentParser): Argument parser to extend.

    Command-line arguments:
        --target: Select the X-HEEP execution target, such as a simulator,
            board, or FPGA target.
        --linker: Select the program memory layout and its corresponding
            linker script.
    """
    parser.set_defaults(toolchain_install_dir = "/app/install/riscv", target = "sim", linker = "on_chip")

    parser.add_argument(
        "--target",
        dest = "target",
        metavar = "TARGET",
        choices = (
            "sim",
            "systemc",
            "nexys-a7-100t",
            "pynq-z2",
            "zcu102",
            "zcu104",
            "genesys2",
            "aup-zu3",
            "vpk180",
        ),
        action = CMakeCacheVariableAction,
        cmake_variable = "XHEEP_TARGET",
        help = "Generated X-HEEP software target.",
    )

    parser.add_argument(
        "--linker",
        dest = "linker",
        metavar = "LAYOUT",
        choices = ("on_chip", "flash_load", "flash_exec"),
        action = CMakeCacheVariableAction,
        cmake_variable = "XHEEP_LINKER",
        help = "Linker Corresponding to the memory layout of your target",
    )


if __name__ == "__main__":
    sys.exit(
        main(default_platform = "Xheep",
             default_simulator = "verilator",
             tiling_enabled = False,
             parser_setup_callback = setup_xheep_defaults))
