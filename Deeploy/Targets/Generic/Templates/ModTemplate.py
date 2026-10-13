from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ModTemplate(NodeTemplate):

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


referenceTemplate = _ModTemplate("""
// Mod (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${A_type.typeName} ref_${C}_${A} = ${A};
${B_type.typeName} ref_${C}_${B} = ${B};
${C_type.typeName} ref_${C}_${C} = ${C};

int32_t inputA_${C}[${shapeA_len}]=${shapeA};
int32_t inputB_${C}[${shapeB_len}]=${shapeB};

Mod_s${A_type.referencedType.typeWidth}_s${B_type.referencedType.typeWidth}(
                ref_${C}_${A},${A_offset},
                ref_${C}_${B},${B_offset},
                0,
                ref_${C}_${C},${C_offset},
                inputA_${C}, ${shapeA_len},
                inputB_${C}, ${shapeB_len});

END_SINGLE_CORE
""")
