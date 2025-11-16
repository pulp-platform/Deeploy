# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity


class ProfilingCodeGeneration(CodeTransformationPass):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addLeft(NodeTemplate("""
        uint32_t ${op}_cycles = getCycles();
        """), {"op": name})
        executionBlock.addRight(
            NodeTemplate("""
        uint32_t ${op}_endCycles = getCycles();
        printf("${op} took %u cycles \\n", ${op}_endCycles - ${op}_cycles);
        """), {"op": name})
        return ctxt, executionBlock
