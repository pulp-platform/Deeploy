# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":
    parser = TestRunnerArgumentParser(
        tiling_arguments = False,
        description = "Deeploy Code Generation Utility for the Single Cluster SoftHier (no Tiling).")

    parser.add_argument('--num_clusters',
                        metavar = 'num_clusters',
                        dest = 'num_clusters',
                        type = int,
                        default = 1,
                        help = 'Number of clusters\n')

    parser.add_argument('--verbose', metavar = 'verbose', dest = 'verbose', type = int, default = 2, help = 'verbose\n')

    for action in parser._actions:
        if action.dest == 'toolchain_install_dir':
            action.default = "${SOFTHIER_INSTALL_DIR}/third_party/toolchain/install"
    args = parser.parse_args()

    testRunner = TestRunner(platform = "SoftHier", simulator = "gvsoc", tiling = False, argument_parser = parser)

    testRunner.cmake_args += f" -D num_clusters={args.num_clusters}"

    testRunner.run()
