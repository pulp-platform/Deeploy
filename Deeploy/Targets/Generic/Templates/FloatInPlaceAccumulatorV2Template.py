# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer


class _GenericInPlaceAccumulatorV2Template(NodeTemplate):
    """True in-place InPlaceAccumulatorV2 template (generic / single-core).

    Writes the result directly into accum_buffer rather than into a separate
    data_out buffer.  data_out is registered as an alias of accum_buffer so
    the memory allocator treats them as sharing the same physical memory.

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
        assert isinstance(accum_buffer, VariableBuffer)
        assert isinstance(data_out, VariableBuffer)
        # Bidirectional alias: data_out shares memory with accum_buffer.
        accum_buffer.aliases.add(data_out.name)
        data_out.aliases.add(accum_buffer.name)
        return ctxt, operatorRepresentation, []


referenceTemplate = _GenericInPlaceAccumulatorV2Template("""
// InPlaceAccumulatorV2 - true in-place (Name: ${nodeName}, Op: ${nodeOp})
// data_out aliases accum_buffer (same memory); result written directly to accum_buffer.
// Reset (lazy_reset_grad=1): accum_buffer  = gradient
// Accum (lazy_reset_grad=0): accum_buffer += gradient
BEGIN_SINGLE_CORE
    if (${lazy_reset_grad}[0]) {
        for (uint32_t i = 0; i < ${size}; i++) {
            ${accum_buffer}[i] = ${gradient}[i];
        }
    } else {
        for (uint32_t i = 0; i < ${size}; i++) {
            ${accum_buffer}[i] += ${gradient}[i];
        }
    }
END_SINGLE_CORE
""")
