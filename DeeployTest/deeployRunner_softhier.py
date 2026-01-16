#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Deeploy runner for SoftHier platform."""

import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":

    # Define parser setup callback to add SoftHier-specific arguments
    def setup_parser(parser):
        parser.add_argument('--num-clusters',
                            type = int,
                            default = 1,
                            dest = 'num_clusters',
                            help = 'Number of clusters (default: 1)\n')
        parser.add_argument('--cores', type = int, default = 8, help = 'Number of cores (default: 8)\n')

    sys.exit(
        main(default_platform = "SoftHier",
             default_simulator = "gvsoc",
             tiling_enabled = False,
             parser_setup_callback = setup_parser))
