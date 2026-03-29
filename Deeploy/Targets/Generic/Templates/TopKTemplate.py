from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


referenceTemplate = NodeTemplate("""
// TopK (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
// Find the top 10 values and their indices
// Assumes 1D input for simplicity
typedef struct {
	${data_in_type.referencedType.typeName} value;
	uint32_t index;
} topk_pair_t;

topk_pair_t pairs[${data_in_size}];
for (uint32_t i = 0; i < ${data_in_size}; ++i) {
	pairs[i].value = ((${data_in_type.referencedType.typeName}*)${data_in})[i];
	pairs[i].index = i;
}
// Simple selection sort for top-k
for (uint32_t i = 0; i < 10; ++i) {
	uint32_t max_idx = i;
	for (uint32_t j = i + 1; j < ${data_in_size}; ++j) {
		if (pairs[j].value > pairs[max_idx].value) {
			max_idx = j;
		}
	}
	// Swap
	if (max_idx != i) {
		topk_pair_t tmp = pairs[i];
		pairs[i] = pairs[max_idx];
		pairs[max_idx] = tmp;
	}
	// Write output
	((${values_out_type.referencedType.typeName}*)${values_out})[i] = pairs[i].value;
	((${indices_out_type.referencedType.typeName}*)${indices_out})[i] = pairs[i].index;
}
END_SINGLE_CORE
""")