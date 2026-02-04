# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Tuple

from Deeploy.DeeployTypes import _NoVerbosity
from Deeploy.DeeployTypes import CodeGenVerbosity
from Deeploy.DeeployTypes import CodeTransformationPass
from Deeploy.DeeployTypes import ExecutionBlock
from Deeploy.DeeployTypes import NetworkContext
from Deeploy.DeeployTypes import NodeTemplate


class SnitchCoreFilterPass(CodeTransformationPass):

    def __init__(self, coreType: Literal["dm", "compute"]):
        super().__init__()
        self.coreType = coreType

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addLeft(NodeTemplate(f"if (snrt_is_{self.coreType}_core()) {{\n"), {})
        executionBlock.addRight(NodeTemplate("}\n"), {})
        return ctxt, executionBlock
