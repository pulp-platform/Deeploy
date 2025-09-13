# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Transpose ${data_in_shape} -> ${data_out_shape} (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
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
END_SINGLE_CORE
""")
