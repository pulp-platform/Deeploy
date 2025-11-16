# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity

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
