# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatAddTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Always populate strides/out_shape/ndim. The unified Add_fp32 kernel
        # detects identity strides at runtime and falls through to a simple
        # element-wise loop, so non-broadcast adds pay only the upfront
        # ndim-step stride check (a few cycles per call).
        if 'strides1' not in operatorRepresentation:
            out_shape = list(ctxt.lookup(operatorRepresentation['data_out']).shape)
            natural_strides = []
            stride = 1
            for dim in reversed(out_shape):
                natural_strides.insert(0, stride)
                stride *= dim
            operatorRepresentation['ndim'] = len(out_shape)
            operatorRepresentation['out_shape'] = out_shape
            operatorRepresentation['strides1'] = natural_strides
            operatorRepresentation['strides2'] = natural_strides

        operatorRepresentation['strides1_str'] = '{' + ', '.join(map(str, operatorRepresentation['strides1'])) + '}'
        operatorRepresentation['strides2_str'] = '{' + ', '.join(map(str, operatorRepresentation['strides2'])) + '}'
        operatorRepresentation['out_shape_str'] = '{' + ', '.join(map(str, operatorRepresentation['out_shape'])) + '}'

        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatAddTemplate("""
// Snitch FP32 Add (Name: ${nodeName}, Op: ${nodeOp})
{
    uint32_t strides1[${ndim}] = ${strides1_str};
    uint32_t strides2[${ndim}] = ${strides2_str};
    uint32_t out_shape[${ndim}] = ${out_shape_str};
    Add_fp32_broadcast(${data_in_1}, ${data_in_2}, ${data_out}, out_shape, strides1, strides2, ${ndim}, ${size});
}
""")
