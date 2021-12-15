# ----------------------------------------------------------------------
#
# File: iRMSNormTemplate.py
#
# Last edited: 20.02.2024
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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _iRMSNormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        return ctxt, operatorRepresentation, []


referenceTemplate = _iRMSNormTemplate("""
// iRMSnorm (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE iRMSnorm_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_plp(${data_in}, ${data_out}, ${weight}, ${size}, ${lastDimLength}, ${log2D});
""")
