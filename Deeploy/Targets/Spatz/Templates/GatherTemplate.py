from Deeploy.DeeployTypes import NodeTemplate

memcpyTemplate = NodeTemplate("""
// Gather (Name: ${nodeName}, Op: ${nodeOp})
<%
width = int(data_in_type.referencedType.typeWidth/8)
%>

% if num_indices == 1:
for (uint32_t i=0; i<${batch}; ++i) {
    memcpy(${data_out} + i * ${axis_length}, ${data_in} + i * ${batch_length} + ${index} * ${axis_length}, ${axis_length} * ${width});
}
% elif batch==1:
for (uint32_t j=0; j<${num_indices}; ++j) {
    memcpy(${data_out} + j * ${axis_length}, 
            ${data_in} + ${indices}[j] * ${axis_length}, 
            ${axis_length} * ${width});
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
""")


memcpyDualCoreTemplate = NodeTemplate("""
// Gather (Name: ${nodeName}, Op: ${nodeOp})
<%
width = int(data_in_type.referencedType.typeWidth/8)
%>
const unsigned int cid = snrt_cluster_core_idx();
% if num_indices == 1:
for (uint32_t i=0; i<${batch}; ++i) {
    memcpy(${data_out} + i * ${axis_length}, ${data_in} + i * ${batch_length} + ${index} * ${axis_length}, ${axis_length} * ${width});
}
% elif batch==1:
if (cid == 0){
    for (uint32_t j=0; j<${num_indices // 2}; ++j) {
        memcpy(${data_out} + j * ${axis_length}, 
                ${data_in} + ${indices}[j] * ${axis_length}, 
                ${axis_length} * ${width});
    }
} else {
    for (uint32_t j=${num_indices // 2}; j<${num_indices}; ++j) {
        memcpy(${data_out} + j * ${axis_length}, 
                ${data_in} + ${indices}[j] * ${axis_length}, 
                ${axis_length} * ${width});
    }
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
""")