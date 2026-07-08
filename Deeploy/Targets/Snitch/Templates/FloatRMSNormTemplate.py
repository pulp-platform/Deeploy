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

        operatorRepresentation["inputSize"] = int(np.prod(input_shape))
        operatorRepresentation["lastDimLength"] = operatorRepresentation["NormalizedAxesSize"]

        return ctxt, operatorRepresentation, []


FloatRMSNormTemplateStr = r"""
RMSNorm_fp32(${data_in}, ${weight}, ${data_out}, ${inputSize}, ${lastDimLength}, ${eps});
"""

referenceTemplate = FloatRMSNormTemplate(FloatRMSNormTemplateStr)

# SSR + FREP variant: sum-of-squares reduction streams via SSR + FREP register
# accumulate (no DM2 write stream); scale/output stays a normal-store loop.
# Requires operands in TCDM/L1 (tiled flow).
FloatRMSNormSSRTemplateStr = r"""
RMSNorm_fp32_ssr_frep(${data_in}, ${weight}, ${data_out}, ${inputSize}, ${lastDimLength}, ${eps});
"""

ssrFrepTemplate = FloatRMSNormTemplate(FloatRMSNormSSRTemplateStr)
