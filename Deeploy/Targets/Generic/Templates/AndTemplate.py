from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _AndTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in_1 = ctxt.lookup(operatorRepresentation['data_in_1'])
        data_in_2 = ctxt.lookup(operatorRepresentation['data_in_2'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        if hasattr(data_in_1, "_signed") and hasattr(data_in_1, "nLevels"):
            operatorRepresentation['data_in_1_offset'] = (data_in_1._signed == 0) * int(data_in_1.nLevels / 2)
        else:
            operatorRepresentation['data_in_1_offset'] = 0

        if hasattr(data_in_2, "_signed") and hasattr(data_in_2, "nLevels"):
            operatorRepresentation['data_in_2_offset'] = (data_in_2._signed == 0) * int(data_in_2.nLevels / 2)
        else:
            operatorRepresentation['data_in_2_offset'] = 0

        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)
        else:
            operatorRepresentation['output_offset'] = 0

        return ctxt, operatorRepresentation, []


referenceTemplate = _AndTemplate("""
// Add (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i=0;i<${size};i++){
        ${data_out}[i] = ((${data_in_1}[i] + ${data_in_1_offset}) && (${data_in_2}[i] + ${data_in_2_offset})) + ${output_offset};
    }
END_SINGLE_CORE
""")
