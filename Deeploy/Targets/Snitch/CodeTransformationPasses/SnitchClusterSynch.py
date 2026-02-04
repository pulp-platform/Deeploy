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

_synchTemplate = NodeTemplate("""
        snrt_cluster_hw_barrier();
""")


class SnitchSynchCoresPass(CodeTransformationPass):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addLeft(_synchTemplate, {})
        executionBlock.addRight(_synchTemplate, {})
        return ctxt, executionBlock
