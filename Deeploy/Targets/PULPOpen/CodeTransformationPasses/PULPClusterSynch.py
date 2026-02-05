# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity

_synchTemplate = NodeTemplate("""
        pi_cl_team_barrier();
        """)


class PULPSynchCoresPass(CodeTransformationPass):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addRight(_synchTemplate, {})
        return ctxt, executionBlock
