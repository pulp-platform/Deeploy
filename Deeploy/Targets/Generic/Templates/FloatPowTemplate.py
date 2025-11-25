# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _PowTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Get input and output tensors
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        # Get data type (fp32 or fp16)
        data_type = data_in._type.typeName
        operatorRepresentation['data_type'] = data_type

        # Exponent must be a constant integer
        if 'exponent' in operatorRepresentation:
            exponent_input = operatorRepresentation['exponent']
            if isinstance(exponent_input, str):
                # It's a tensor name - not supported for integer exponent version
                raise ValueError("Tensor exponent not supported. Use constant integer exponent.")
            else:
                # Convert to integer
                operatorRepresentation['exponent_value'] = int(exponent_input)

        # Calculate size
        operatorRepresentation['size'] = int(np.prod(data_in.shape))

        return ctxt, operatorRepresentation, []


referenceTemplate = _PowTemplate("""
// Pow (Name: ${nodeName}, Op: ${nodeOp})
% if 'float32' in data_type:
Pow_fp32_int32_fp32(${data_in}, ${exponent_value}, ${data_out}, ${size});
% elif 'float16' in data_type:
Pow_fp16_int32_fp16(${data_in}, ${exponent_value}, ${data_out}, ${size});
% endif
""")
