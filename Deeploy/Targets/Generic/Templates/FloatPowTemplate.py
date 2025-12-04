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

        # Get type width dynamically (e.g., 32, 64)
        type_width = data_in._type.referencedType.typeWidth
        operatorRepresentation['type_width'] = type_width

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
            # Since currently the kernel only supports equally sized base-exponent data,
            # for non-scalar, let's add a size check here (length of data_in should be equal to exponent length).
            if input_size != exponent_size:
                raise ValueError(f"Pow operator mismatch: input size ({input_size}) "
                                 f"must equal exponent size ({exponent_size}) for non-scalar exponents.")

            operatorRepresentation['is_scalar'] = False
            operatorRepresentation['exponent_scalar'] = "NULL"

        return ctxt, operatorRepresentation, []


referenceTemplate = _PowTemplate("""
// Pow (Name: ${nodeName}, Op: ${nodeOp})
% if is_scalar:
Pow_fp${type_width}_scalar_fp${type_width}(${data_in}, ${exponent_scalar}, ${data_out}, ${size});
% else:
Pow_fp${type_width}_fp${type_width}_fp${type_width}(${data_in}, ${exponent}, ${data_out}, ${size});
% endif
""")
