# ----------------------------------------------------------------------
#
# File: ConcatTemplate.py
#
# Last edited: 19.02.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
