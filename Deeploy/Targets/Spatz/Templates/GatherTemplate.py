from Deeploy.DeeployTypes import NodeTemplate

# TODO for l3 -> l3 transfers in spatz should memcpy be used?
referenceTemplate = NodeTemplate("""
// Gather (Name: ${nodeName}, Op: ${nodeOp})
<%
width = int(data_in_type.referencedType.typeWidth/8)
%>
BEGIN_SINGLE_CORE
% if num_indices == 1:
for (uint32_t i=0; i<${batch}; ++i) {
    snrt_dma_start_1d(${data_out} + i * ${axis_length}, ${data_in} + i * ${batch_length} + ${index} * ${axis_length}, ${axis_length} * ${width});
}
% elif batch==1:
for (uint32_t j=0; j<${num_indices}; ++j) {
    snrt_dma_start_1d(${data_out} + j * ${axis_length}, 
            ${data_in} + ${indices}[j] * ${axis_length}, 
            ${axis_length} * ${width});
}
% else:
for (uint32_t i=0; i<${batch}; ++i) {
    for (uint32_t j=0; j<${num_indices}; ++j) {
        snrt_dma_start_1d(${data_out} + i * (${num_indices} * ${axis_length}) + j * ${axis_length}, 
               ${data_in} + i * ${batch_length} + ${indices}[j] * ${axis_length}, 
               ${axis_length} * ${width});
    }
}
% endif
END_SINGLE_CORE
""")
