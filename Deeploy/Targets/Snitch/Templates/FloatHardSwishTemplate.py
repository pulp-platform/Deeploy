# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.DeeployTypes import OperatorRepresentation


class FloatHardSwishTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation["data_in"])
        operatorRepresentation["size"] = int(np.prod(data_in.shape))

        return ctxt, operatorRepresentation, []


FloatHardSwishTemplateStr = r"""
HardSwish_fp32(${data_in}, ${data_out}, ${size});
"""

referenceTemplate = FloatHardSwishTemplate(FloatHardSwishTemplateStr)
