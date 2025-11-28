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
        exponent = ctxt.lookup(operatorRepresentation['exponent'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        
        # Get data type (fp32)
        data_type = data_in._type.typeName
        operatorRepresentation['data_type'] = data_type
        
        # Calculate size
        input_size = int(np.prod(data_in.shape))
        exponent_size = int(np.prod(exponent.shape))
        operatorRepresentation['size'] = input_size
        
        # Check if exponent is scalar (broadcasting)
        if exponent_size == 1:
            operatorRepresentation['is_scalar'] = True
            # Get the full variable name with prefix
            exponent_name = operatorRepresentation['exponent']
            operatorRepresentation['exponent_scalar'] = f"DeeployNetwork_{exponent_name}[0]"
        else:
            operatorRepresentation['is_scalar'] = False
        
        return ctxt, operatorRepresentation, []

referenceTemplate = _PowTemplate("""
// Pow (Name: ${nodeName}, Op: ${nodeOp})
% if is_scalar:
Pow_fp32_scalar_fp32(${data_in}, ${exponent_scalar}, ${data_out}, ${size});
% else:
Pow_fp32_fp32_fp32(${data_in}, ${exponent}, ${data_out}, ${size});
% endif
""")
