# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.CommonExtensions.CodeTransformationPasses.CycleMeasurement import ProfilingCodeGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity


class PULPProfileUntiled(CodeTransformationPass):

    def __init__(self):
        self.profileUntiled = ProfilingCodeGeneration()

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        # Also activate under tilingProfiling so a single --profileTiling run
        # gets whole-node "took N cycles" markers in addition to per-tile beacons.
        if verbose.untiledProfiling or verbose.tilingProfiling:
            ctxt, executionBlock = self.profileUntiled.apply(ctxt, executionBlock, name)

        return ctxt, executionBlock


class PULPNodeBeacon(CodeTransformationPass):
    """Outermost per-node entry beacon.

    Placed as the *last* pass in a transformer so its ``addLeft`` statement is the
    frontmost code emitted for the node -- before the inter-node L3 memory
    management (cl_ram_malloc/free) and any tiling/closure setup. This is the only
    beacon that survives a hang in the L3 allocation/free that happens *between*
    nodes, which is invisible to PULPProfileUntiled (inside the L3 mem-management
    wrap) and to the per-tile beacons. The last "[NODE-ENTER]" on the UART pins
    the frozen node regardless of how it was lowered (tiled, untiled, or skip).
    """

    _nodeEnterBeacon = NodeTemplate("""
    if (pi_core_id() == 0) { printf("[NODE-ENTER] ${nodeName}\\r\\n"); }
    """)

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        if verbose.untiledProfiling or verbose.tilingProfiling:
            executionBlock.addLeft(self._nodeEnterBeacon, {"nodeName": name})

        return ctxt, executionBlock
