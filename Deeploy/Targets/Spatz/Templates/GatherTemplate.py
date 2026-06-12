# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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

dynamicDMAtemplate = NodeTemplate("""
// Gather (Name: ${nodeName}, Op: ${nodeOp})
// Dynamic DMA strategy (Spatz):
// - indices already transferred to local memory by the tiling pass
// - fetch selected rows directly from external data_in into local data_out
<%
width = int(data_in_type.referencedType.typeWidth/8)
%>

// Currently supported configuration: axis=0 and batch=1 (matches existing Spatz Gather tests)
if ((${axis} != 0) || (${batch} != 1)) {
    error();
} else {
    if (snrt_is_dm_core()) {
        const size_t bytes_per_row = (size_t)${axis_length} * (size_t)${width}; // sizeof(${data_in_type.referencedType.typeName}) = ${width}
        char *dst_base = (char *)${data_out};
        const char *src_base = (const char *)${data_in};

        for (size_t j = 0; j < (size_t)${num_indices}; ++j) {
            const size_t dst_off = j * bytes_per_row;
            const size_t src_off = (size_t)${indices}[j] * bytes_per_row;
            snrt_dma_start_1d((void *)(dst_base + dst_off), (const void *)(src_base + src_off), bytes_per_row);
        }

        // Ensure all row DMAs complete before the tiling pass starts the output transfer.
        snrt_dma_wait_all();
    }
}
""")