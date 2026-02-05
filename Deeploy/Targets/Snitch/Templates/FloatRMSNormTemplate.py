# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class FloatRMSNormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation["data_in"])
        input_shape = list(data_in.shape)

        axis = operatorRepresentation.get("axis", -1)
        if axis < 0:
            axis = len(input_shape) + axis

        operatorRepresentation["lastDimLength"] = data_in.shape[-1]
        operatorRepresentation["size"] = int(np.prod(input_shape))
        operatorRepresentation["inputSize"] = int(np.prod(input_shape))
        operatorRepresentation["NormalizedAxesSize"] = int(np.prod(input_shape[axis:]))

        return ctxt, operatorRepresentation, []


FloatRMSNormTemplateStr = r"""
RMSNorm_fp32(${data_in}, ${weight}, ${data_out}, ${size}, ${lastDimLength}, ${eps});
"""

referenceTemplate = FloatRMSNormTemplate(FloatRMSNormTemplateStr)
