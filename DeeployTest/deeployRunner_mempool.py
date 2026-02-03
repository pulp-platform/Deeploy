#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Deeploy runner for MemPool platform."""

import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":

    # Define parser setup callback to add MemPool-specific arguments
    def setup_parser(parser):
        parser.add_argument('--num-cores',
                            type = int,
                            default = 16,
                            dest = 'num_cores',
                            help = 'Number of cores (default: 16)\n')

    sys.exit(
        main(default_platform = "MemPool",
             default_simulator = "banshee",
             tiling_enabled = False,
             parser_setup_callback = setup_parser))
