# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":
    parser = TestRunnerArgumentParser(
        tiling_arguments = False, description = "Deeploy Code Generation Utility for the MemPool Platform (no Tiling).")

    parser.add_argument('-n',
                        metavar = 'num_threads',
                        dest = 'num_threads',
                        type = int,
                        default = 16,
                        help = 'Number of parallel threads\n')
    args = parser.parse_args()

    testRunner = TestRunner(platform = "MemPool", simulator = "banshee", tiling = False, argument_parser = parser)

    testRunner.cmake_args += f" -D num_threads={args.num_threads}"

    testRunner.run()
