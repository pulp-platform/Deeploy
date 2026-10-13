from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatWhereTemplate(NodeTemplate):  #dont know how to creat BOOL_T data type

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        indice = ctxt.lookup(operatorRepresentation['cond'])
        operatorRepresentation['cond_offset'] = 0

        if hasattr(indice, "_signed") and hasattr(indice, "nLevels"):
            operatorRepresentation['cond_offset'] = (indice._signed == 0) * int(indice.nLevels / 2)
        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatWhereTemplate("""
// Float Where (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${cond_type.typeName} ref_${data_out}_${cond} = ${cond};
${B_type.typeName} ref_${data_out}_${B} = ${B};
${C_type.typeName} ref_${data_out}_${C} = ${C};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

int32_t inputA_${data_out}[${shapecond_len}]=${shapecond};
int32_t inputB_${data_out}[${shapeB_len}]=${shapeB};
int32_t inputC_${data_out}[${shapeC_len}]=${shapeC};

Where_fp${B_type.referencedType.typeWidth}_fp${C_type.referencedType.typeWidth}(
                ref_${data_out}_${cond},${cond_offset},
                ref_${data_out}_${B},
                ref_${data_out}_${C},
                inputA_${data_out}, ${shapecond_len},
                inputB_${data_out}, ${shapeB_len},
                inputC_${data_out}, ${shapeC_len},
                ref_${data_out}_${data_out} );

END_SINGLE_CORE
""")
