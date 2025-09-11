# ----------------------------------------------------------------------
#
# File: RqGemmTemplate.py
#
# Last edited: 03.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class SnitchRqGemmTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        if isinstance(operatorRepresentation['alpha'], float):
            assert operatorRepresentation['alpha'].is_integer()
            operatorRepresentation['alpha'] = int(operatorRepresentation['alpha'])
        if isinstance(operatorRepresentation['beta'], float):
            assert operatorRepresentation['beta'].is_integer()
            operatorRepresentation['beta'] = int(operatorRepresentation['beta'])

        #LMACAN: WARNING: Assumes rounding is expected
        add = ctxt.lookup(operatorRepresentation['add'])
        add.values += 2**(operatorRepresentation['log2D'] - 1)

        if operatorRepresentation['transB']:
            operatorRepresentation['kernelName'] = 'RQGemm_s8_transB_row_parallel_unrolled'
        else:
            operatorRepresentation['kernelName'] = 'RQGemm_s8_row_parallel_unrolled'

        return ctxt, operatorRepresentation, []


SnitchRqGemmTemplateStr = r"""
${kernelName}(${A}, ${B}, ${C}, ${data_out}, ${M}, ${N}, ${O}, ${alpha}, ${beta}, ${mul}, ${add}, ${log2D});
"""

SnitchRqGemm_Template = SnitchRqGemmTemplate(SnitchRqGemmTemplateStr)
