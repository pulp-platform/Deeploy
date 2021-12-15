# ----------------------------------------------------------------------
#
# File: ReshapeTemplate.py
#
# Last edited: 16.12.2021
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


class _ReshapeTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # SCHEREMO: Selectively mark 'indices' dead, since we don't need them
        if 'indices' in operatorRepresentation.keys():
            ctxt.globalObjects[operatorRepresentation['indices']]._deploy = False
            ctxt.globalObjects[operatorRepresentation['indices']]._live = False

        inBuffer = ctxt.lookup(operatorRepresentation['data_in'])
        outBuffer = ctxt.lookup(operatorRepresentation['data_out'])
        outBuffer._alias = inBuffer.name

        return ctxt, operatorRepresentation, []


referenceTemplate = _ReshapeTemplate("""
// Reshape (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE ${data_out} = ${data_in};
""")
