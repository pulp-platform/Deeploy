# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = False,
        description = "Deeploy Code Generation Utility for the Chimera Platform (Host, no Tiling).")

    parser.add_argument('--simulator',
                        metavar = "<simulator>",
                        dest = "simulator",
                        type = str,
                        choices = ["gvsoc"],
                        default = "gvsoc",
                        help = "Select the simulator to use")

    args = parser.parse_args()

    testRunner = TestRunner(platform = "Chimera", simulator = args.simulator, tiling = False, argument_parser = parser)

    testRunner.run()
