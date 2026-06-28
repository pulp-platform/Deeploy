# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _PULPInPlaceAccumulatorV2Template(NodeTemplate):
    """True in-place InPlaceAccumulatorV2 template for PULP.

    Writes the accumulation result into ``accum_buffer`` (the graph input).
    ``data_out`` is registered as an alias of ``accum_buffer`` so the memory
    allocator knows they share memory and will not free ``accum_buffer``
    prematurely.

    ``data_out`` is intentionally *not* written by the emitted C code:

    - InPlaceAccumulatorV2 is terminal in the training graph — no downstream
      kernel consumes ``data_out``; it only exists as a symbolic output so
      the graph stays well-formed.
    - In the tiled path, emitting a write to ``data_out`` would also make
      Deeploy generate an L2 egress DMA for it, and ``data_out``'s L2 slot
      may overlap with other live buffers, corrupting L2.

    Semantics:
        if lazy_reset_grad: accum_buffer  = gradient   (reset)
        else:               accum_buffer += gradient   (accumulate)
    """

    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        accum_buffer = ctxt.lookup(operatorRepresentation['accum_buffer'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        # data_out is a symbolic in-place output aliased to accum_buffer (which the
        # kernel accumulates into directly). The bidirectional `aliases` set keeps
        # both live; no tiler-level `_alias` is needed.
        accum_buffer.aliases.add(data_out.name)
        data_out.aliases.add(accum_buffer.name)

        return ctxt, operatorRepresentation, []


referenceTemplate = _PULPInPlaceAccumulatorV2Template("""
// InPlaceAccumulatorV2 (Name: ${nodeName}, Op: ${nodeOp})
// Writes result into accum_buffer (in-place).  data_out is an alias of
// accum_buffer and is deliberately not written — it has no downstream
// consumer, and emitting a write would trigger an L2 egress DMA whose
// destination may overlap with live buffers in the tiled path.
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
