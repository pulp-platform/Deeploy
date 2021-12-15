# ----------------------------------------------------------------------
#
# File: testRunner_mempool.py
#
# Last edited: 17.03.2023
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
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
        tiling_arguments = False, description = "Deeploy Code Generation Utility for the MemPool Platform (no Tiling).")

    parser.add_argument('-n',
                        metavar = 'num_threads',
                        dest = 'num_threads',
                        type = int,
                        default = 16,
                        help = 'Number of parallel threads\n')
    args = parser.parse_args()

    testRunner = TestRunner(platform = "MemPool", simulator = "banshee", tiling = False, argument_parser = parser)

    testRunner.cmake_args += f" -D num_threads={args.num_threads}"

    testRunner.run()
