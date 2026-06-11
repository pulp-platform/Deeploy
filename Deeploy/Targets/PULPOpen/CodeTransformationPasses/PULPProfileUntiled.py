# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.CommonExtensions.CodeTransformationPasses.CycleMeasurement import ProfilingCodeGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity


class PULPProfileUntiled(CodeTransformationPass):

    # Whole-node entry beacon. This pass wraps every node's execution block --
    # tiled or untiled -- so it is the one place that catches a hang in an
    # *untiled* node (which never enters the L3/cluster tiling codegen and is
    # therefore invisible to the per-tile beacons). The last "[NODE-ENTER]" on
    # the UART pinpoints the frozen node regardless of how it was lowered.
    _nodeEnterBeacon = NodeTemplate("""
    if (pi_core_id() == 0) { printf("[NODE-ENTER] ${nodeName}\\r\\n"); }
    """)

    def __init__(self):
        self.profileUntiled = ProfilingCodeGeneration()

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        # Also activate under tilingProfiling so a single --profileTiling run
        # gets whole-node entry/exit markers in addition to per-tile beacons.
        if verbose.untiledProfiling or verbose.tilingProfiling:
            ctxt, executionBlock = self.profileUntiled.apply(ctxt, executionBlock, name)
            # addLeft last => frontmost statement of the node, printed before the
            # getCycles wrap and before any tiling setup/alloc.
            executionBlock.addLeft(self._nodeEnterBeacon, {"nodeName": name})

        return ctxt, executionBlock
