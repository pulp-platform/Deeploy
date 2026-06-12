from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


selectionSortTemplate = NodeTemplate("""
// TopK node: finds the top ${k_value} values and their indices
// Assumes 1D input 
${data_in_type.referencedType.typeName} *values_tmp = snrt_l1alloc(sizeof(${data_in_type.referencedType.typeName})*${data_in_size});
${indices_out_type.referencedType.typeName} *indices_tmp = snrt_l1alloc(sizeof(${indices_out_type.referencedType.typeName})*${data_in_size});

for (uint32_t i = 0; i < ${data_in_size}; ++i) {
	values_tmp[i] = ((${data_in_type.referencedType.typeName}*)${data_in})[i];
	indices_tmp[i] = i;
}
// Simple selection sort for top-k
for (uint32_t i = 0; i < ${k_value}; ++i) {
	uint32_t max_idx = i;
	for (uint32_t j = i + 1; j < ${data_in_size}; ++j) {
        if (values_tmp[j] > values_tmp[max_idx]) {
          max_idx = j;
        }
	}
	// Swap
	if (max_idx != i) {
		float32_t tmp_val = values_tmp[i];
		int32_t tmp_idx = indices_tmp[i];
		values_tmp[i] = values_tmp[max_idx];
		indices_tmp[i] = indices_tmp[max_idx];
		values_tmp[max_idx] = tmp_val;
		indices_tmp[max_idx] = tmp_idx;
	}
	// Write output
	((${values_out_type.referencedType.typeName}*)${values_out})[i] = values_tmp[i];
	((${indices_out_type.referencedType.typeName}*)${indices_out})[i] = indices_tmp[i];
}
""")

# compute_topk_vector_instructions
minHeapTemplate = NodeTemplate("""
compute_topk_min_heap(
    ${k_value},
    ${data_in_size},
    ${data_in},
    ${values_out},
    ${indices_out}
);

""")