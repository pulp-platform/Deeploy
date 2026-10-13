from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _CastTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # SCHEREMO: Selectively mark 'indices' dead, since we don't need them
        if 'indices' in operatorRepresentation.keys():
            ctxt.globalObjects[operatorRepresentation['indices']]._deploy = False
            ctxt.globalObjects[operatorRepresentation['indices']]._live = False

        # Same for "shape"
        if "shape" in operatorRepresentation.keys():
            ctxt.globalObjects[operatorRepresentation["shape"]]._deploy = False
            ctxt.globalObjects[operatorRepresentation["shape"]]._live = False

        inBuffer = ctxt.lookup(operatorRepresentation['data_in'])
        outBuffer = ctxt.lookup(operatorRepresentation['data_out'])
        outBuffer._alias = inBuffer.name

        #in case of a type needing offset
        A = ctxt.lookup(operatorRepresentation['data_in'])
        C = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['A_offset'] = 0
        operatorRepresentation['C_offset'] = 0
        if hasattr(A, "_signed") and hasattr(A, "nLevels"):
            operatorRepresentation['A_offset'] = (A._signed == 0) * int(A.nLevels / 2)
        if hasattr(C, "_signed") and hasattr(C, "nLevels"):  #TypeChecker when output is unsigned, input is signed
            operatorRepresentation['C_offset'] = -(C._signed == 0) * int(C.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _CastTemplate("""
// Cast (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE

for (uint32_t i=0;i<${size};i++){
    ${data_out}[i] = (${data_out_type.referencedType.typeName}) (${data_in}[i]+${A_offset}) + ${C_offset};
}
END_SINGLE_CORE
""")
