from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


#B is ALwAYS signed, so we ALWAYS have to compensate the offset, even in the float case
class _FloatGatherElementsTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        indice = ctxt.lookup(operatorRepresentation['B'])
        operatorRepresentation['B_offset'] = 0

        if hasattr(indice, "_signed") and hasattr(indice, "nLevels"):
            operatorRepresentation['B_offset'] = (indice._signed == 0) * int(indice.nLevels / 2)
        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatGatherElementsTemplate("""
// Float GatherElements (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${A_type.typeName} ref_${C}_${A} = ${A};
${B_type.typeName} ref_${C}_${B} = ${B};
${C_type.typeName} ref_${C}_${C} = ${C};

int32_t inputA_${C}[${shapeA_len}]=${shapeA};
int32_t inputB_${C}[${shapeB_len}]=${shapeB};

GatherElements_fp${A_type.referencedType.typeWidth}_s${B_type.referencedType.typeWidth}(
                ref_${C}_${A},
                inputA_${C}, ${shapeA_len},
                ref_${C}_${B},
                inputB_${C},${shapeB_len},${B_offset},
                ${axis},
                ref_${C}_${C}
                );

END_SINGLE_CORE
""")
