#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.deeployRunner import main
import sys

if __name__ == "__main__":

    # Define parser setup callback to add Snitch-specific arguments
    def setup_parser(parser):
        parser.add_argument('--num-cores',
                            type = int,
                            default = 8,
                            dest = 'num_cores',
                            help = 'Number of cores (default: 8)\n')

    sys.exit(
        main(default_platform = "Snitch",
             default_simulator = "gvsoc",
             tiling_enabled = True,
             parser_setup_callback = setup_parser))
