from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ComparisonTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['C'])

        input_1_offset = 0
        if hasattr(A, "_signed") and hasattr(A, "nLevels"):
            input_1_offset = (A._signed == 0) * int(A.nLevels / 2)
        input_2_offset = 0
        if hasattr(B, "_signed") and hasattr(B, "nLevels"):
            input_2_offset = (B._signed == 0) * int(B.nLevels / 2)

        output_offset = 0
        if hasattr(C, "_signed") and hasattr(C, "nLevels"):
            output_offset = -(C._signed == 0) * int(C.nLevels // 2)

        operatorRepresentation['offset'] = input_1_offset - input_2_offset + output_offset

        return ctxt, operatorRepresentation, []


referenceTemplate = _ComparisonTemplate("""
<%
    index_A = 0 if A_scalar else "i"
    index_B = 0 if B_scalar else "i"
%>
// Greater (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${A_type.typeName} ref_${C}_${A} = ${A};
${B_type.typeName} ref_${C}_${B} = ${B};
${C_type.typeName} ref_${C}_${C} = ${C};


for(uint32_t i=0; i<${size}; i++){
    ref_${C}_${C}[i] = ((ref_${C}_${A}[${index_A}] - ref_${C}_${B}[${index_B}] + ${offset}) ${op} 0);
}

END_SINGLE_CORE
""")
