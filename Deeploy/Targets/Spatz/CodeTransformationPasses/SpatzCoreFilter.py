# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity


class SpatzCoreFilterPass(CodeTransformationPass):

    def __init__(self, coreType: Literal["dm", "compute"]):
        super().__init__()
        self.coreType = coreType

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        theother = self.coreType=="dm" ? "compute" : "dm"
        executionBlock.addLeft(NodeTemplate(f"if (snrt_is_{theother}_core()) {{\n"), {})
        executionBlock.addRight(NodeTemplate("}\n"), {})
        return ctxt, executionBlock
