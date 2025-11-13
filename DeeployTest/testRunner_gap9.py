# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = False, description = "Deeploy Code Generation Utility for the GAP9 Platform (no Tiling).")

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

    # Set default GVSOC install dir
    for action in parser._actions:
        if action.dest == 'gvsoc_install_dir':
            action.default = "${GAP_SDK_HOME}/install/workstation"
    args = parser.parse_args()

    testRunner = TestRunner(platform = "GAP9", simulator = "gvsoc", tiling = False, argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"
    testRunner.run()
