# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _SqrtTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Get input and output tensors
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        # Get data type (fp32 or fp16)
        data_type = data_in._type.typeName
        operatorRepresentation['data_type'] = data_type

        # Calculate size
        operatorRepresentation['size'] = int(np.prod(data_in.shape))

        return ctxt, operatorRepresentation, []


referenceTemplate = _SqrtTemplate("""
// Sqrt (Name: ${nodeName}, Op: ${nodeOp})
% if 'float32' in data_type:
Sqrt_fp32_fp32(${data_in}, ${data_out}, ${size});
% elif 'float16' in data_type:
Sqrt_fp16_fp16(${data_in}, ${data_out}, ${size});
% endif
""")
