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

        # C template uses ${size} and ${lastDimLength}
        operatorRepresentation["size"] = int(np.prod(input_shape))
        operatorRepresentation["lastDimLength"] = operatorRepresentation["NormalizedAxesSize"]

        return ctxt, operatorRepresentation, []


FloatRMSNormTemplateStr = r"""
RMSNorm_fp32(${data_in}, ${weight}, ${data_out}, ${size}, ${lastDimLength}, ${eps});
"""

referenceTemplate = FloatRMSNormTemplate(FloatRMSNormTemplateStr)
