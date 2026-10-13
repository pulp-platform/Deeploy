from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


#B is ALwAYS signed, so we ALWAYS have to compensate the offset, even in the float case
class _ExpandTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data = ctxt.lookup(operatorRepresentation['A'])
        indice = ctxt.lookup(operatorRepresentation['B'])
        output = ctxt.lookup(operatorRepresentation['C'])
        operatorRepresentation['A_offset'] = 0
        operatorRepresentation['B_offset'] = 0
        operatorRepresentation['C_offset'] = 0

        if hasattr(data, "_signed") and hasattr(data, "nLevels"):
            operatorRepresentation['A_offset'] = (data._signed == 0) * int(data.nLevels / 2)
        if hasattr(indice, "_signed") and hasattr(indice, "nLevels"):
            operatorRepresentation['B_offset'] = (indice._signed == 0) * int(indice.nLevels / 2)
        if hasattr(output, "_signed") and hasattr(output, "nLevels"):
            operatorRepresentation['C_offset'] = -(output._signed == 0) * int(output.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _ExpandTemplate("""
// Float Expand (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${A_type.typeName} ref_${C}_${A} = ${A};
${B_type.typeName} ref_${C}_${B} = ${B};
${C_type.typeName} ref_${C}_${C} = ${C};

int32_t inputA_${C}[${shapeA_len}]=${shapeA};

Expand_s${A_type.referencedType.typeWidth}(
                ref_${C}_${A},${A_offset},
                ref_${C}_${C},${C_offset},
                ref_${C}_${B},${shapeB_len},${B_offset},
                inputA_${C}, ${shapeA_len}
                );

END_SINGLE_CORE
""")
