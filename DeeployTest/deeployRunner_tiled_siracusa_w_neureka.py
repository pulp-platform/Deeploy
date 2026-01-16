#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.deeployRunner import main
import sys

if __name__ == "__main__":

    # Define parser setup callback to add Siracusa+Neureka-specific arguments
    def setup_parser(parser):
        parser.add_argument('--cores', type = int, default = 8, help = 'Number of cores (default: 8)\n')
        parser.add_argument('--neureka-wmem', action = 'store_true', help = 'Enable Neureka weight memory\n')
        parser.add_argument('--enable-3x3', action = 'store_true', help = 'Enable 3x3 convolutions\n')

    sys.exit(
        main(default_platform = "Siracusa_w_neureka",
             default_simulator = "gvsoc",
             tiling_enabled = True,
             parser_setup_callback = setup_parser))
