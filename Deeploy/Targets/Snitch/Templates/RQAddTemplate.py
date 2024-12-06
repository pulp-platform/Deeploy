# ----------------------------------------------------------------------
#
# File: RQAddTemplate.py
#
# Last edited: 11.11.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurich
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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class SnitchRQAddTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedI2 = ctxt.lookup(operatorRepresentation['data_in_2'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['data_in_1'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['input_2_signed'] = signedI2
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []


referenceTemplate = SnitchRQAddTemplate("""

<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if input_2_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>

// PULP NN RQADD
pulp_nn_add${signatureString}(${data_in_1}, ${data_in_2}, ${data_out}, ${rqs1_mul}, ${rqs1_add}, ${rqs1_log2D}, ${rqs2_mul}, ${rqs2_add}, ${rqs2_log2D}, ${rqsOut_mul}, ${rqsOut_add}, ${rqsOut_log2D}, 1, ${size}, 1, 1);
""")
