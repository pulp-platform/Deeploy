# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = True,
        description = "Deeploy Code Generation Utility for the Siracusa Platform (Tiling & NEureka).")

    parser.add_argument('--cores',
                        metavar = '<cores>',
                        dest = 'cores',
                        type = int,
                        default = 8,
                        help = 'Set number of cluster cores')
    args = parser.parse_args()

    testRunner = TestRunner(platform = "Siracusa", simulator = "gvsoc", tiling = True, argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"

    testRunner.run()
