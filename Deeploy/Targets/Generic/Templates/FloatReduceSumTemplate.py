# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float ReduceSum (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
float32_t ${data_out}_accumulator = 0.0f;
<% reduceLength = 1
for i, axis in enumerate(axes):
    if axis < 0:
        axes[i] += len(data_in_shape)
    reduceLength = reduceLength * data_in_shape[i]
%>
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
${data_out_type.typeName} dummy_${data_out} = ${data_out};

<% restDims = set(list(range(len(data_in_shape)))).difference(set(axes)) %>
% for i in list(restDims):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor

${data_out}_accumulator = 0.0f; // Reset accumulator for each output element

% for i in list(axes):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor

${data_out}_accumulator += ((${data_in_type.referencedType.typeName} (*)${shapeStr})${data_in})${accessStr};

% for i in range(len(axes)):
}
% endfor

*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) (${data_out}_accumulator);

% for i in range(len(restDims)):
}
% endfor
END_SINGLE_CORE
""")
