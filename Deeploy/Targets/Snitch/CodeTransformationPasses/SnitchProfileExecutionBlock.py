# ----------------------------------------------------------------------
#
# File: SnitchProfileExecutionBlock.py
#
# Last edited: 05.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity

_dumpCycleCntTemplate = NodeTemplate("""
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()) {
            #ifndef BANSHEE_SIMULATION
                DUMP(getCycles());
            #else
                printf("${position} of ${nodeName} block at cycle %d \\n", getCycles());
            #endif
        }
""")


class SnitchProfileExecutionBlockPass(CodeTransformationPass):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addLeft(_dumpCycleCntTemplate, {"position": "Start", "nodeName": name})
        executionBlock.addRight(_dumpCycleCntTemplate, {"position": "End", "nodeName": name})
        return ctxt, executionBlock
