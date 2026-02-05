# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ITAMaxTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        return ctxt, operatorRepresentation, []

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # WIESEP: Hack: Allocate a buffer for each core
        size = operatorRepresentation['lastDimLength'] * 192
        name = operatorRepresentation['nodeName'] + f"_buffer"
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size

        return ctxt, operatorRepresentation, [name]


MemPoolParallelTemplate = _ITAMaxTemplate("""
// ITAMax Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);

ITAMax_parallel_s${data_in_type.referencedType.typeWidth}(
    ${data_in},
    ${data_out},
    ${ctxtBuffer},
    ${size},
    ${lastDimLength},
    ${n_levels},
    core_id,
    numThreads
);
mempool_barrier(numThreads);
""")
