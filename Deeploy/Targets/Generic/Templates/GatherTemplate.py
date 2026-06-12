# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Gather (Name: ${nodeName}, Op: ${nodeOp})
<%
width = int(data_in_type.referencedType.typeWidth/8)
%>
BEGIN_SINGLE_CORE
% if num_indices == 1:
for (uint32_t i=0; i<${batch}; ++i) {
    memcpy(${data_out} + i * ${axis_length}, ${data_in} + i * ${batch_length} + ${index} * ${axis_length}, ${axis_length} * ${width});
}
% else:
for (uint32_t i=0; i<${batch}; ++i) {
    for (uint32_t j=0; j<${num_indices}; ++j) {
        memcpy(${data_out} + i * (${num_indices} * ${axis_length}) + j * ${axis_length},
               ${data_in} + i * ${batch_length} + ${indices}[j] * ${axis_length},
               ${axis_length} * ${width});
    }
}
% endif
END_SINGLE_CORE
""")
