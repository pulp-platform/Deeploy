# ----------------------------------------------------------------------
#
# File: iSoftmaxPreAllocatedBuffTemplate.py
#
# Last edited: 09.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, scheremo@iis.ee.ethz.ch, ETH Zurich
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

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class iSoftmaxPreAllocatedBuffTemplate(NodeTemplate):

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:

        lastDimBuffer_dim = 8 * 4 * operatorRepresentation['lastDimLength']
        lastDimBuffer_name = operatorRepresentation['nodeName'] + "_lastDimBuffer"
        return [(lastDimBuffer_name, lastDimBuffer_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        lastDimBuffer_name, lastDimBuffer_dim = iSoftmaxPreAllocatedBuffTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(lastDimBuffer_name, lastDimBuffer_dim)

        operatorRepresentation['lastDimBuffer'] = lastDimBuffer_name
        operatorRepresentation['lastDimBufferSize'] = lastDimBuffer_dim
        return ctxt, operatorRepresentation, [lastDimBuffer_name]

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0

        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []
