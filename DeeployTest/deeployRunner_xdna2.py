#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper that invokes the shared Deeploy test runner for the XDNA2 platform.

Usage (from DeeployTest/):
    python deeployRunner_xdna2.py -t Tests/Kernels/BF16/Add/Regular [--skipsim] [-v]
"""

import sys

from testUtils.deeployRunner import main

if __name__ == '__main__':
    sys.exit(main(default_platform = "XDNA2", default_simulator = "host", tiling_enabled = True))
