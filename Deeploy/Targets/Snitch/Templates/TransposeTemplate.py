# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Use snrt_cluster_core_idx() == 0 instead of BEGIN_SINGLE_CORE macro to avoid core_id dependency
referenceTemplate = NodeTemplate("""
// Transpose ${data_in_shape} -> ${data_out_shape} (Name: ${nodeName}, Op: ${nodeOp})
if (snrt_cluster_core_idx() == 0) {
${data_out_type.typeName} dummy_${data_out} = ${data_out};
<%
    dimStr = ''
    accessStr = ''
    shapeStr = ''
    for dim in data_in_shape:
        dimStr += '['+str(dim)+']'
%>
% for idx, i in enumerate(perm[:-1]):
<%
    shapeStr += '['+str(data_in_shape[idx+1])+']'
%>
% endfor
% for idx, i in enumerate(perm):
<%
    shape = data_out_shape[idx]
    accessStr += '[i_'+str(idx)+']'
%>
for(uint32_t i_${i} = 0; i_${i}<${shape}; i_${i}++){
% endfor
*dummy_${data_out}++ = ((${data_in_type.referencedType.typeName} (*)${shapeStr})${data_in})${accessStr};
% for idx, i in enumerate(perm):
}
% endfor
}
""")
