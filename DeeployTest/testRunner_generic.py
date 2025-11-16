# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = False,
        description = "Deeploy Code Generation Utility for the Generic Platform (Host Machine, no Tiling).")
    args = parser.parse_args()

    testRunner = TestRunner(platform = "Generic", simulator = "host", tiling = False, argument_parser = parser)

    testRunner.run()
