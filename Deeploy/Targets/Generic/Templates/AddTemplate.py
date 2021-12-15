# ----------------------------------------------------------------------
#
# File: AddTemplate.py
#
# Last edited: 15.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _AddTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in_1 = ctxt.lookup(operatorRepresentation['data_in_1'])
        data_in_2 = ctxt.lookup(operatorRepresentation['data_in_2'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        input_1_offset = 0
        if hasattr(data_in_1, "_signed") and hasattr(data_in_1, "nLevels"):
            input_1_offset = (data_in_1._signed == 0) * int(data_in_1.nLevels / 2)
        input_2_offset = 0
        if hasattr(data_in_2, "_signed") and hasattr(data_in_2, "nLevels"):
            input_2_offset = (data_in_2._signed == 0) * int(data_in_2.nLevels / 2)
        output_offset = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            output_offset = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        operatorRepresentation['offset'] = input_1_offset + input_2_offset + output_offset

        return ctxt, operatorRepresentation, []


referenceTemplate = _AddTemplate("""
// Add (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i=0;i<${size};i++){
        ${data_out}[i] = ${data_in_1}[i] + ${data_in_2}[i] + ${offset};
    }
END_SINGLE_CORE
""")
