from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


#B is ALwAYS signed, so we ALWAYS have to compensate the offset, even in the float case
class _FloatSplitTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        indice = ctxt.lookup(operatorRepresentation['B'])
        operatorRepresentation['B_offset'] = 0

        if hasattr(indice, "_signed") and hasattr(indice, "nLevels"):
            operatorRepresentation['B_offset'] = (indice._signed == 0) * int(indice.nLevels / 2)
        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatSplitTemplate("""
// Float Split (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
<%

test = [C0,C1,C2]
test_type=[C0_type,C1_type,C2_type]
all=[]
all_type=[]
all_type_len=[]
for i in range(num_output):
    all.append(test[i])
    all_type.append(test_type[i].typeName)
    all_type_len.append(test_type[i].referencedType.typeWidth//8)

%>
// Deeploy only ever allocates the first output. hence, we need to deeploy_malloc all the other ones for multiple outputs nodes
// "name = (type.typeName) deeploy_malloc(type.referencedType.typeWidth//8 * size);"
% for i in range(1, num_output) :
${all[i]} = (${all_type[i]}) deeploy_malloc(${all_type_len[i]} * ${size});

% endfor

//printf(\" Consequences\\n\");
${A_type.typeName} ref_${C0}_${A} = ${A};
${B_type.typeName} ref_${C0}_${B} = ${B};

${C0_type.typeName} all_ref_${C0}[3]={NULL,NULL,NULL};//array of ptr
//printf(\" Its not the NULL array\\n\");
//Mako Template to handle any number of output
//from Slice Template
//for axis in (list(preAxes) + list(axes)):
//uint32_t data_out  _offset_  axis  = 0;
//endfor

% for i in range(num_output) :
all_ref_${C0}[${i}]= ${all[i]};
// printf(\" Got the pointer %p \\n\", all_ref_${C0}[${i}]);
% endfor
// printf(\" ITS not the assignements\\n\");
int32_t inputA_${C0}[${shapeA_len}]=${shapeA};
// printf(\" It can only be the function call\\n\");
Split_fp${A_type.referencedType.typeWidth}(
                ref_${C0}_${A},inputA_${C0}, ${shapeA_len},
                ref_${C0}_${B}, ${sizeB},${B_offset},
                ${axis},
                all_ref_${C0}, ${num_output}
                );
// printf(\" Of the made up Split\\n\");
END_SINGLE_CORE
""")
