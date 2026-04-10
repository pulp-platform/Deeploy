#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
CLI runner for training tests on Siracusa and GAP9.

Usage:
    python deeployTrainingRunner.py -t <test_dir> [-p Siracusa|GAP9] [--tiled] [options]

Examples:
    python deeployTrainingRunner.py -t Tests/Models/MLP_Train/simplemlp_train
    python deeployTrainingRunner.py -t Tests/Models/MLP_Train/simplemlp_train -p GAP9
    python deeployTrainingRunner.py -t Tests/Models/SmallTransformer/tinytransformer_train --tiled
    python deeployTrainingRunner.py -t Tests/Models/SmallTransformer/tinytransformer_train --tiled -p GAP9
"""

import argparse
import sys

from testUtils.deeployTrainingRunner import main

if __name__ == '__main__':
    # Peek at --tiled and -p before passing to main(), which builds its own parser.
    pre = argparse.ArgumentParser(add_help = False)
    pre.add_argument('--tiled', action = 'store_true', default = False)
    pre.add_argument('-p', '--platform', default = 'Siracusa')
    known, _ = pre.parse_known_args()

    sys.exit(main(tiling_enabled = known.tiled, default_platform = known.platform))
