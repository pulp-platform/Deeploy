# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# Deeploy/Targets/Snitch/Templates/FloatAddTemplate.py

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatAddTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Always initialize these variables to avoid Mako errors
        operatorRepresentation.setdefault('need_broadcast', False)
        operatorRepresentation.setdefault('ndim', 0)
        operatorRepresentation.setdefault('strides1_str', '{}')
        operatorRepresentation.setdefault('strides2_str', '{}')
        operatorRepresentation.setdefault('out_shape_str', '{}')

        # If broadcasting is required, generate the stride array strings
        if operatorRepresentation['need_broadcast']:
            strides1 = operatorRepresentation['strides1']
            strides2 = operatorRepresentation['strides2']
            out_shape = operatorRepresentation['out_shape']
            operatorRepresentation['strides1_str'] = '{' + ', '.join(map(str, strides1)) + '}'
            operatorRepresentation['strides2_str'] = '{' + ', '.join(map(str, strides2)) + '}'
            operatorRepresentation['out_shape_str'] = '{' + ', '.join(map(str, out_shape)) + '}'

        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatAddTemplate("""
// Snitch FP32 Add (Name: ${nodeName}, Op: ${nodeOp})
% if need_broadcast:
{
    uint32_t strides1[${ndim}] = ${strides1_str};
    uint32_t strides2[${ndim}] = ${strides2_str};
    uint32_t out_shape[${ndim}] = ${out_shape_str};
    Add_fp32_broadcast(${data_in_1}, ${data_in_2}, ${data_out}, out_shape, strides1, strides2, ${ndim}, ${size});
}
% else:
Add_fp32(${data_in_1}, ${data_in_2}, ${data_out}, ${size});
% endif
""")
