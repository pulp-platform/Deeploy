# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatReduceMeanTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatReduceMeanTemplate("""
## =============== Compute required variables ===============
## Compute the total number of elements being reduced in one axis
<%
reduceLength = 1

for i, axis in enumerate(axes):
    if axis < 0:
        axes[i] += len(data_in_shape)
    reduceLength = reduceLength * data_in_shape[axis]
%>

## Compute the remaining dimensions after reduction
<%
restDims = set(list(range(len(data_in_shape)))).difference(set(axes))
%>

## =============== Prepare shape and access strings ===============
## shapeStr is going to have the [d1][d2]... format
## accessStr is going to have the [i_0][i_1]... format
<%
    shapeStr = ''
    accessStr = ''
%>

% for idx, i in enumerate(data_in_shape[1:]):
<%
    shapeStr += '['+str(i)+']'
%>
% endfor

% for j in range(len(data_in_shape)):
<%
    accessStr += '[i_'+str(j)+']'
%>
% endfor

## =============== Start of the actual template ===============
// ReduceMean (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${data_out_type.referencedType.typeName} ${data_out}_accumulator = 0;
${data_out_type.typeName} dummy_${data_out} = ${data_out};

% for i in list(restDims):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator = ${input_offset}*${reduceLength};

% for i in list(axes):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator += ((${data_in_type.referencedType.typeName} (*)${shapeStr})${data_in})${accessStr};
% for i in range(len(axes)):
}
% endfor

% if keepdims:
*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) (${data_out}_accumulator / ${reduceLength} + ${output_offset});
% else:
*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) (${data_out}_accumulator / ${reduceLength});
% endif
% for i in range(len(restDims)):
}
% endfor

END_SINGLE_CORE
""")
