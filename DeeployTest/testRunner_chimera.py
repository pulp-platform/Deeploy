# ----------------------------------------------------------------------
#
# File: testRunner_chimera.py
#
# Last edited: 16.06.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Victor Jung, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

if __name__ == "__main__":

    parser = TestRunnerArgumentParser(
        tiling_arguments = False,
        description = "Deeploy Code Generation Utility for the Chimera Platform (Host, no Tiling).")
    args = parser.parse_args()

    testRunner = TestRunner(platform = "Chimera", simulator = "host", tiling = False, argument_parser = parser)

    testRunner.run()
