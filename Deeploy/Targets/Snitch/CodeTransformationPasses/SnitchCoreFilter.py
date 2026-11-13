# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity


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
        if self.coreType == "compute":
            # The FPU is decoupled from the integer core: wait until all issued
            # FP instructions (e.g. a trailing fdiv + store) have completed, so
            # the following barrier / DMA transfer sees the final results.
            executionBlock.addRight(NodeTemplate("snrt_fpu_fence();\n}\n"), {})
        else:
            executionBlock.addRight(NodeTemplate("}\n"), {})
        return ctxt, executionBlock
