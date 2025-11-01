# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = False,
        description = "Deeploy Code Generation Utility for the Siracusa Platform (no Tiling).")

    parser.add_argument('--cores',
                        metavar = '<cores>',
                        dest = 'cores',
                        type = int,
                        default = 8,
                        help = 'Set number of cluster cores')

    parser.add_argument('--profileUntiled',
                        action = 'store_true',
                        dest = 'profileUntiled',
                        default = False,
                        help = 'Profile Untiled')

    args = parser.parse_args()

    testRunner = TestRunner(platform = "Siracusa", simulator = "gvsoc", tiling = False, argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"
    testRunner.run()
