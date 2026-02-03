# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(tiling_arguments = True,
                                      description = "Deeploy Code Generation Utility for the Snitch Platform (Tiling).")

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

    testRunner = TestRunner(platform = "Snitch", simulator = args.simulator, tiling = True, argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"

    testRunner.run()
