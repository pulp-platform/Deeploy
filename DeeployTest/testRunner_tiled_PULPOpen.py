# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = True, description = "Deeploy Code Generation Utility for the Pulp Open Platform (Tiling).")

    parser.add_argument('--cores',
                        metavar = '<cores>',
                        dest = 'cores',
                        type = int,
                        default = 8,
                        help = 'Set number of cluster cores')
    parser.add_argument('--simulator',
                        dest = 'simulator',
                        default = 'gvsoc',
                        choices = ['gvsoc', 'banshee', 'qemu', 'vsim', 'qsim.gui', 'qsim', 'vsim.gui', 'host', 'none'],
                        help = 'set simulator')
    args = parser.parse_args()

    testRunner = TestRunner(platform = "PULPOpen", simulator = args.simulator, tiling = True, argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"

    testRunner.run()