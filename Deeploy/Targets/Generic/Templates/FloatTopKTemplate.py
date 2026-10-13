from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


#B is ALwAYS signed, so we ALWAYS have to compensate the offset, even in the float case
class _FloatTopKTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        valK = ctxt.lookup(operatorRepresentation['K'])
        operatorRepresentation['K_offset'] = 0

        if hasattr(valK, "_signed") and hasattr(valK, "nLevels"):
            operatorRepresentation['K_offset'] = (valK._signed == 0) * int(valK.nLevels / 2)

        indice = ctxt.lookup(operatorRepresentation['I'])
        operatorRepresentation['I_offset'] = 0
        if hasattr(indice, "_signed") and hasattr(indice, "nLevels"):
            if indice._signed != None:  #error when last layer of a network, indice. are None thus preventing this
                operatorRepresentation['I_offset'] = -(indice._signed == 0) * int(indice.nLevels / 2)
        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatTopKTemplate("""
// Float TopK (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
// printf(\"Trying to TopK with output Indices ${I_type.referencedType.typeWidth}/8 * ${I_size} \\n\");

${I} = (${I_type.typeName}) deeploy_malloc(${I_type.referencedType.typeWidth}/8 * ${I_size});

${A_type.typeName} ref_${C}_${A} = ${A};
${K_type.typeName} ref_${C}_${K} = ${K};
${C_type.typeName} ref_${C}_${C} = ${C};
${I_type.typeName} ref_${C}_${I} = ${I};

int32_t inputA_${C}[${shapeA_len}]=${shapeA};

TopK_fp${A_type.referencedType.typeWidth}(
                ref_${C}_${A},
                ref_${C}_${C},
                ref_${C}_${I},${I_offset},
                ref_${C}_${K},${K_offset},
                ${axis},
                inputA_${C}, ${shapeA_len}
                );
// printf(\"TopK was not the problem...\\n\");

END_SINGLE_CORE
""")
