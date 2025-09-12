# ----------------------------------------------------------------------
#
# File: __init__.py
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


class SnitchGemmTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        if isinstance(operatorRepresentation['alpha'], float):
            assert operatorRepresentation['alpha'].is_integer(
            ), f"Parameter alpha is not an integer: {operatorRepresentation['alpha']}"
            operatorRepresentation['alpha'] = int(operatorRepresentation['alpha'])
        if isinstance(operatorRepresentation['beta'], float):
            assert operatorRepresentation['beta'].is_integer(
            ), f"Parameter beta is not an integer: {operatorRepresentation['beta']}"
            operatorRepresentation['beta'] = int(operatorRepresentation['beta'])

        if operatorRepresentation['transB']:
            operatorRepresentation['kernelName'] = "Gemm_s8_transB_row_parallel"
        else:
            operatorRepresentation['kernelName'] = "Gemm_s8_row_parallel"

        return ctxt, operatorRepresentation, []


SnitchGemmTemplateStr = r"""
${kernelName}(${A}, ${B}, ${C}, ${data_out}, ${M}, ${N}, ${O}, ${alpha}, ${beta});
"""

SnitchGemm_Template = SnitchGemmTemplate(SnitchGemmTemplateStr)
