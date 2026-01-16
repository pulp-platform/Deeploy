#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


import sys
from testUtils.deeployRunner import main


if __name__ == "__main__":
    sys.exit(main(default_platform="QEMU-ARM", default_simulator="qemu", tiling_enabled=False))
