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
        size = operatorRepresentation['lastDimLength']
        name = operatorRepresentation['nodeName'] + f"_buffer"
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size

        return ctxt, operatorRepresentation, [name]


referenceTemplate = _ITAMaxTemplate("""
// ITAMax (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE ITAMax_s${data_in_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${ctxtBuffer}, ${size}, ${lastDimLength}, ${n_levels});
""")
