#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":

    # Define parser setup callback to add GAP9+NE16-specific arguments
    def setup_parser(parser):
        parser.add_argument('--cores', type = int, default = 8, help = 'Number of cores (default: 8)\n')
        parser.add_argument('--ne16-wmem', action = 'store_true', help = 'Enable NE16 weight memory\n')
        parser.add_argument('--enable-3x3', action = 'store_true', help = 'Enable 3x3 convolutions\n')

    sys.exit(
        main(default_platform = "GAP9_w_NE16",
             default_simulator = "gvsoc",
             tiling_enabled = True,
             parser_setup_callback = setup_parser))
