# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = False,
        description = "Deeploy Code Generation Utility for the Magia Platform (no Tiling).")

    parser.add_argument('--tiles',
                        metavar = '<tiles>',
                        dest = 'tiles',
                        type = int,
                        default = 4,
                        help = 'Set number of mesh tiles')
    
    args = parser.parse_args()

    testRunner = TestRunner(platform = "Magia", simulator = "none", tiling = False, argument_parser = parser)

    testRunner.run()
