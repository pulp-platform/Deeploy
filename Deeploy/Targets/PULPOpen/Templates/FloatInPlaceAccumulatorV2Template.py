# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer


class _PULPInPlaceAccumulatorV2Template(NodeTemplate):
    """True in-place InPlaceAccumulatorV2 template for PULP.

    Writes the result directly into accum_buffer (the graph input) rather
    than into a separate data_out buffer.  data_out is registered as an
    alias of accum_buffer so the memory allocator knows they share memory
    and will not free accum_buffer prematurely.

    Semantics:
        if lazy_reset_grad: accum_buffer = gradient        (reset)
        else:               accum_buffer += gradient       (accumulate)
    """

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        accum_buffer = ctxt.lookup(operatorRepresentation['accum_buffer'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        accum_buffer.aliases.add(data_out.name)
        data_out.aliases.add(accum_buffer.name)
        return ctxt, operatorRepresentation, []


referenceTemplate = _PULPInPlaceAccumulatorV2Template("""
// InPlaceAccumulatorV2 - true in-place (Name: ${nodeName}, Op: ${nodeOp})
// Writes result to accum_buffer (in-place) and data_out (explicit output).
// In training, data_out aliases accum_buffer (same or separate allocation).
// Reset (lazy_reset_grad=1): accum_buffer  = gradient
// Accum (lazy_reset_grad=0): accum_buffer += gradient
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
int32_t ${nodeName}_start = MIN(${nodeName}_chunk * ${nodeName}_core_id, (int32_t)${size});
int32_t ${nodeName}_stop  = MIN(${nodeName}_start + ${nodeName}_chunk,   (int32_t)${size});

if (${lazy_reset_grad}[0]) {
    for (int32_t i = ${nodeName}_start; i < ${nodeName}_stop; i++) {
        ${accum_buffer}[i] = ${gradient}[i];
        ${data_out}[i] = ${gradient}[i];
    }
} else {
    for (int32_t i = ${nodeName}_start; i < ${nodeName}_stop; i++) {
        ${accum_buffer}[i] += ${gradient}[i];
        ${data_out}[i] = ${accum_buffer}[i];
    }
}
""")

# Tiled variant: writes only to ${accum_buffer} (no ${data_out} write).
# In the tiled context the optimizer reads the gradient directly from
# accum_buffer's L2 address (input_4/input_5).  data_out's L2 address may
# overlap with other live buffers, so writing to it via DMA would corrupt L2.
# Omitting ${data_out} means we do not need a DMA egress for it at all.
tiledReferenceTemplate = _PULPInPlaceAccumulatorV2Template("""
// InPlaceAccumulatorV2 - tiled in-place (Name: ${nodeName}, Op: ${nodeOp})
// Tiled variant: result written only to accum_buffer (egressed to L2 by DMA).
// data_out is NOT written here — optimizer reads gradient from accum_buffer.
// Reset (lazy_reset_grad=1): accum_buffer  = gradient
// Accum (lazy_reset_grad=0): accum_buffer += gradient
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
int32_t ${nodeName}_start = MIN(${nodeName}_chunk * ${nodeName}_core_id, (int32_t)${size});
int32_t ${nodeName}_stop  = MIN(${nodeName}_start + ${nodeName}_chunk,   (int32_t)${size});

if (${lazy_reset_grad}[0]) {
    for (int32_t i = ${nodeName}_start; i < ${nodeName}_stop; i++) {
        ${accum_buffer}[i] = ${gradient}[i];
    }
} else {
    for (int32_t i = ${nodeName}_start; i < ${nodeName}_stop; i++) {
        ${accum_buffer}[i] += ${gradient}[i];
    }
}
""")
