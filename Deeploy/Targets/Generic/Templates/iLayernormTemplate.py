# ----------------------------------------------------------------------
#
# File: ILayernormTemplate.py
#
# Last edited: 31.12.2021
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

from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _iLayerNormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _iLayerNormTemplate("""
// iLayernorm (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Layernorm_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${weight}, ${bias}, ${input_offset}, ${size}, ${lastDimLength}, ${log2D});
""")
