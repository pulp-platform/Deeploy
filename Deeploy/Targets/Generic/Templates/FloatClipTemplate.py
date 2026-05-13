# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ClipTemplate(NodeTemplate):

    def alignToContext(self, ctxt, operatorRepresentation):
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['size'] = int(np.prod(data_in.shape))
        return ctxt, operatorRepresentation, []


referenceTemplate = _ClipTemplate("""
// Clip (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i = 0; i < ${size}; i++){
        ${data_out}[i] = fmaxf(${min_val}f, fminf(${max_val}f, ${data_in}[i]));
    }
END_SINGLE_CORE
""")
