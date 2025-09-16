# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = False,
        description = "Deeploy Code Generation Utility for the ARM (QEMU) Platform (no Tiling).")
    args = parser.parse_args()

    testRunner = TestRunner(platform = "QEMU-ARM", simulator = "qemu", tiling = False, argument_parser = parser)

    testRunner.run()
