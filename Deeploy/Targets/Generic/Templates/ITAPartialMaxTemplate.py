# ----------------------------------------------------------------------
#
# File: ITAPartialMaxTemplate.py
#
# Last edited: 08.01.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
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


class _ITAPartialMaxTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()
        return ctxt, operatorRepresentation, []


referenceTemplate = _ITAPartialMaxTemplate("""
// ITAPartialMax (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE ITAPartialMax_s${data_in_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${size}, ${lastDimLength}, ${group_width}, ${n_levels});
""")
