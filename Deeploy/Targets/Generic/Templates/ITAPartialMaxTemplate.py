# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ITAPartialMaxTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()
        return ctxt, operatorRepresentation, []


referenceTemplate = _ITAPartialMaxTemplate("""
// ITAPartialMax (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE ITAPartialMax_s${data_in_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${size}, ${lastDimLength}, ${group_width}, ${n_levels});
""")
