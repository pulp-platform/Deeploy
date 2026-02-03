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

        # Set default GVSOC install dir for GAP9
        for action in parser._actions:
            if action.dest == 'gvsoc_install_dir':
                action.default = "${GAP_SDK_HOME}/install/workstation"

    sys.exit(
        main(default_platform = "GAP9",
             default_simulator = "gvsoc",
             tiling_enabled = True,
             parser_setup_callback = setup_parser))
