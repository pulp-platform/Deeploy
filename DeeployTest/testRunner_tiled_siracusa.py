# ----------------------------------------------------------------------
#
# File: testRunner_tiled_siracusa.py
#
# Last edited: 31.10.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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
        tiling_arguments = True,
        description = "Deeploy Code Generation Utility for the Siracusa Platform (Tiling & NEureka).")

    parser.add_argument('--cores',
                        metavar = '<cores>',
                        dest = 'cores',
                        type = int,
                        default = 1,
                        help = 'Set number of cluster cores')
    args = parser.parse_args()

    testRunner = TestRunner(platform = "Siracusa", simulator = "gvsoc", tiling = True, argument_parser = parser)

    testRunner.cmake_args += f" -D NUM_CORES={args.cores}"

    testRunner.run()
