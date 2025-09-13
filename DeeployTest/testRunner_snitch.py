# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = False, description = "Deeploy Code Generation Utility for the Snitch Platform (no Tiling).")

    parser.add_argument('--cores',
                        metavar = '<cores>',
                        dest = 'cores',
                        type = int,
                        default = 9,
                        help = 'Set number of cluster cores')
    parser.add_argument('--simulator',
                        metavar = "<simulator>",
                        dest = "simulator",
                        type = str,
                        choices = ["gvsoc", "banshee", "vsim", "vsim.gui"],
                        default = "gvsoc",
                        help = "Select the simulator to use")
    args = parser.parse_args()

    testRunner = TestRunner(platform = "Snitch", simulator = args.simulator, tiling = False, argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"
    testRunner.run()
