# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ConcatTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        dataIn1 = ctxt.lookup(operatorRepresentation['data_in_1'])
        dataIn2 = ctxt.lookup(operatorRepresentation['data_in_2'])

        assert "data_in_3" not in operatorRepresentation.keys(), "Concat with more than two inputs not implemented!"

        dataIn1Shape = dataIn1.shape
        dataIn2Shape = dataIn2.shape

        axis = operatorRepresentation['axis']
        in1TransferLength = np.prod(dataIn1Shape[axis:]) * (dataIn1._type.referencedType.typeWidth // 8)
        in2TransferLength = np.prod(dataIn2Shape[axis:]) * (dataIn2._type.referencedType.typeWidth // 8)

        iterations1 = np.prod(dataIn1Shape[:axis])
        iterations2 = np.prod(dataIn2Shape[:axis])

        assert iterations1 == iterations2, f"iterations1 {iterations1} is not iterations2 {iterations2}; concat can't be applied!"

        operatorRepresentation['iterations'] = iterations1
        operatorRepresentation['in1TransferLength'] = in1TransferLength
        operatorRepresentation['in2TransferLength'] = in2TransferLength

        return ctxt, operatorRepresentation, []


referenceTemplate = _ConcatTemplate("""

char* ${data_in_1}_tf = (char*) ${data_in_1};
char* ${data_in_2}_tf = (char*) ${data_in_2};
char* ${data_out}_tf = (char*) ${data_out};

for (int i=0; i<${iterations}; i++){
memcpy(${data_out}_tf, ${data_in_1}_tf, ${in1TransferLength});
${data_out}_tf += ${in1TransferLength};
${data_in_1}_tf += ${in1TransferLength};
memcpy(${data_out}_tf, ${data_in_2}_tf, ${in2TransferLength});
${data_out}_tf += ${in2TransferLength};
${data_in_2}_tf += ${in2TransferLength};
}
""")
