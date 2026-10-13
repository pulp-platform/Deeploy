from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ScatterNDTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        indice = ctxt.lookup(operatorRepresentation['indice'])
        operatorRepresentation['indice_offset'] = 0

        if hasattr(indice, "_signed") and hasattr(indice, "nLevels"):
            operatorRepresentation['indice_offset'] = (indice._signed == 0) * int(indice.nLevels / 2)
        return ctxt, operatorRepresentation, []


referenceTemplate = _ScatterNDTemplate("""
// ScatterND (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${data_type.typeName} ref_${data_out}_${data} = ${data};
${indice_type.typeName} ref_${data_out}_${indice} = ${indice}; //MUST BE int64_t

${updates_type.typeName} ref_${data_out}_${updates} = ${updates};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};


const int64_t ${data_out}_dims[${k}]=${d};


ScatterND_s${data_type.referencedType.typeWidth}(
ref_${data_out}_${data} ,
ref_${data_out}_${indice}, ${indice_offset},
ref_${data_out}_${updates},
ref_${data_out}_${data_out},
${data_out}_dims,
${n_idx}, ${n_kr},
${k}, ${data_size}
);
//printf(\"SCattering worked \\n\\n\");
END_SINGLE_CORE
""")
