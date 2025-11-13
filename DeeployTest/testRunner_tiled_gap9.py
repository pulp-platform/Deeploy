# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(tiling_arguments = True,
                                      description = "Deeploy Code Generation Utility for the GAP9 Platform (Tiling).")

    parser.add_argument('--cores',
                        metavar = '<cores>',
                        dest = 'cores',
                        type = int,
                        default = 8,
                        help = 'Set number of cluster cores')

    args = parser.parse_args()

    testRunner = TestRunner(platform = "GAP9", simulator = "gvsoc", tiling = True, argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"
    testRunner.run()
