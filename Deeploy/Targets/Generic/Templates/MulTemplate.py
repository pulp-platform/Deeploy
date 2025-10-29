# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.CommonExtensions.NodeTemplate import ElementwiseTemplate
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation


class _MulTemplate(ElementwiseTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['C'])
        operatorRepresentation['A_offset'] = 0
        operatorRepresentation['B_offset'] = 0
        operatorRepresentation['C_offset'] = 0
        if hasattr(A, "_signed") and hasattr(A, "nLevels"):
            operatorRepresentation['A_offset'] = (A._signed == 0) * int(A.nLevels / 2)
        if hasattr(B, "_signed") and hasattr(B, "nLevels"):
            operatorRepresentation['B_offset'] = (B._signed == 0) * int(B.nLevels / 2)
        if hasattr(C, "_signed") and hasattr(C, "nLevels"):
            operatorRepresentation['C_offset'] = -(C._signed == 0) * int(C.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _MulTemplate("""
// Mul (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i=0;i<${size};i++){
        ${C}[i] = ((${A}[i] + ${A_offset}) * (${B}[i] + ${B_offset}) + ${C_offset});
    }
END_SINGLE_CORE
""")
