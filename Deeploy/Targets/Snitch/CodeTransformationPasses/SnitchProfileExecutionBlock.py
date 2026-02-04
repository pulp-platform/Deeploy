# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import _NoVerbosity
from Deeploy.DeeployTypes import CodeGenVerbosity
from Deeploy.DeeployTypes import CodeTransformationPass
from Deeploy.DeeployTypes import ExecutionBlock
from Deeploy.DeeployTypes import NetworkContext
from Deeploy.DeeployTypes import NodeTemplate

_dumpCycleCntTemplate = NodeTemplate("""
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()) {
            #if !defined(BANSHEE_SIMULATION) && !defined(GVSOC_SIMULATION)
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
