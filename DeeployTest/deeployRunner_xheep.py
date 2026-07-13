#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":
    #TODO : make the co-simulation with X-HEEP repository
    sys.exit(main(default_platform = "Xheep", default_simulator = "none", tiling_enabled = False))
