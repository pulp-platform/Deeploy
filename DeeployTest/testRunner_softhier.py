# ----------------------------------------------------------------------
#
# File: testRunner_softhier.py
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author: Bowen Wang <bowwang@iis.ee.ethz.ch> , ETH Zurich
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
        description = "Deeploy Code Generation Utility for the Single Cluster SiftHier (no Tiling).")

    parser.add_argument('--num_clusters',
                        metavar = 'num_clusters',
                        dest = 'num_clusters',
                        type = int,
                        default = 1,
                        help = 'Number of clusters\n')

    parser.add_argument('--verbose', metavar = 'verbose', dest = 'verbose', type = int, default = 2, help = 'verbose\n')
    args = parser.parse_args()

    testRunner = TestRunner(platform = "SoftHier", simulator = "gvsoc", tiling = False, argument_parser = parser)

    testRunner.cmake_args += f" -D num_clusters={args.num_clusters}"

    testRunner.run()
