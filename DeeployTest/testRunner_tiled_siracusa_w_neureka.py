# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
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
    parser.add_argument('--neureka-wmem',
                        dest = "neureka_wmem",
                        action = "store_true",
                        default = False,
                        help = 'Adds weight memory and neureka engine color\n')
    parser.add_argument('--enable-3x3',
                        dest = "enable_3x3",
                        action = "store_true",
                        default = False,
                        help = 'Adds EXPERIMENTAL support for 3x3 convolutions on N-EUREKA\n')
    parser.add_argument('--enableStrides',
                        dest = "enableStrides",
                        action = "store_true",
                        default = False,
                        help = 'Adds EXPERIMENTAL support for strided convolutions on N-EUREKA\n')
    args = parser.parse_args()

    testRunner = TestRunner(platform = "Siracusa_w_neureka",
                            simulator = "gvsoc",
                            tiling = True,
                            argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"
    if args.neureka_wmem:
        testRunner.gen_args += " --neureka-wmem"
    if args.enable_3x3:
        testRunner.gen_args += " --enable-3x3"
    if args.enableStrides:
        testRunner.gen_args += " --enableStrides"

    testRunner.run()
