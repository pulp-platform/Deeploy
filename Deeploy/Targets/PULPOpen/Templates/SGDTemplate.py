# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _PULPSGDTemplate(NodeTemplate):
    """In-place SGD template for PULP.

    weight_updated is aliased to weight so the memory allocator places them
    at the same address in whichever memory level weight lives in (L2 or L3).
    This ensures the tiled egress DMA writes the updated weight back to
    weight's buffer — the same buffer the training network reads from on the
    next forward pass.
    """

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        weight = ctxt.lookup(operatorRepresentation['weight'])
        weight_updated = ctxt.lookup(operatorRepresentation['weight_updated'])

        # Link weight_updated to weight: the bidirectional `aliases` set keeps both
        # live (has_live_aliases), and the allocTemplate override below points
        # weight_updated at weight's storage. No tiler-level `_alias` is needed —
        # the allocTemplate override resolves the aliasing on its own (unlike
        # Reshape, which has no allocTemplate override and does rely on `_alias`).
        weight.aliases.add(weight_updated.name)
        weight_updated.aliases.add(weight.name)

        # weight_updated reuses weight's storage (in whichever memory level weight
        # lives in) rather than getting its own arena slot, so the write lands in
        # weight's buffer — the one the next forward pass reads from.
        weight_updated.allocTemplate = NodeTemplate(" ${name} = (${type.typeName}) " + str(weight._instance) + ";")
        # No deallocTemplate override needed: MemoryAllocation skips the dealloc of
        # any buffer with live aliases, and weight is a live input.
        return ctxt, operatorRepresentation, []


referenceTemplate = _PULPSGDTemplate("""
// SGD Weight Update with Separated Multiplication and Subtraction Unrolling
// (Name: ${nodeName}, Op: ${nodeOp})
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
int32_t ${nodeName}_chunk_start = MIN(${nodeName}_chunk*${nodeName}_core_id, ${size});
int32_t ${nodeName}_chunk_stop = MIN(${nodeName}_chunk_start + ${nodeName}_chunk, ${size});

${weight_type.typeName} ref_${weight} = ${weight};
${grad_type.typeName} ref_${grad} = ${grad};
${weight_type.typeName} ref_${weight_updated} = ${weight_updated};

float32_t learning_rate = ${lr};

// Temporary buffer for multiplication results
float32_t temp_mul[6];

uint32_t i = ${nodeName}_chunk_start;
for (; i+5 < ${nodeName}_chunk_stop; i+=6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_${grad}[i];
    temp_mul[1] = learning_rate * ref_${grad}[i+1];
    temp_mul[2] = learning_rate * ref_${grad}[i+2];
    temp_mul[3] = learning_rate * ref_${grad}[i+3];
    temp_mul[4] = learning_rate * ref_${grad}[i+4];
    temp_mul[5] = learning_rate * ref_${grad}[i+5];

    // Unrolled subtraction operations
    ref_${weight_updated}[i] = ref_${weight}[i] - temp_mul[0];
    ref_${weight_updated}[i+1] = ref_${weight}[i+1] - temp_mul[1];
    ref_${weight_updated}[i+2] = ref_${weight}[i+2] - temp_mul[2];
    ref_${weight_updated}[i+3] = ref_${weight}[i+3] - temp_mul[3];
    ref_${weight_updated}[i+4] = ref_${weight}[i+4] - temp_mul[4];
    ref_${weight_updated}[i+5] = ref_${weight}[i+5] - temp_mul[5];
}

// Handle remaining elements
for (; i < ${nodeName}_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_${grad}[i];
    ref_${weight_updated}[i] = ref_${weight}[i] - temp_grad;
}
""")
