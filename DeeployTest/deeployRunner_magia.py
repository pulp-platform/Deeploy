#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":

    # Define parser setup callback to add Siracusa-specific arguments
    def setup_parser(parser):
        parser.add_argument('--tiles', type = int, default = 4, help = 'Number of mesh tiles (default: 4)')

    sys.exit(
        main(
            default_platform = "Magia",
            default_simulator = "none",
            tiling_enabled = False,
            parser_setup_callback = setup_parser,
        ))
