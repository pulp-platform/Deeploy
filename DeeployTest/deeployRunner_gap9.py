#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":

    # Define parser setup callback to add GAP9-specific arguments
    def setup_parser(parser):
        parser.add_argument('--cores', type = int, default = 8, help = 'Number of cores (default: 8)\n')
        parser.add_argument('--powerMeasurement',
                            action = 'store_true',
                            default = False,
                            help = 'Enable GPIO toggling around the inference window for external '
                            'power measurement (e.g. PPK2). Only meaningful with -s board.\n')

    sys.exit(
        main(default_platform = "GAP9",
             default_simulator = "gvsoc",
             tiling_enabled = False,
             parser_setup_callback = setup_parser))
