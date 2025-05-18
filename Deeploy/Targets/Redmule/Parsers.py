# ----------------------------------------------------------------------
#
# File: BasicParsers.py
#
# Last edited: 15.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Authors:
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

import math
from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Targets.Generic.Parsers import MatMulParser

class GEMMRedmuleParser(MatMulParser):

    def __init__(self, noBiasHoisting = True):
        self.noBiasHoisting = noBiasHoisting
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([
            len(node.inputs) >= 2,
            len(node.outputs) == 1,
            node.attrs['alpha'] == 1
        ])

        if ret:
            if 'transA' in node.attrs:
                self.operatorRepresentation['transA'] = node.attrs['transA']
            else:
                self.operatorRepresentation['transA'] = 0

            if 'transB' in node.attrs:
                self.operatorRepresentation['transB'] = node.attrs['transB']
            else:
                self.operatorRepresentation['transB'] = 0
            if 'alpha' in node.attrs:
                self.operatorRepresentation['alpha'] = node.attrs['alpha']
            else:
                self.operatorRepresentation['alpha'] = 1
            if 'beta' in node.attrs:
                self.operatorRepresentation['beta'] = node.attrs['beta']
            else:
                self.operatorRepresentation['beta'] = 1
        
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['A', 'B']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                if idx < len(inputs):
                    self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            if len(node.inputs) == 3:
                self.operatorRepresentation['C'] = newCtxt.lookup(node.inputs[2].name).name
            elif not self.noBiasHoisting:
                values = np.zeros((1))
                zeroTensor = gs.Constant(f'{node.name}_C_Tensor', values = values)
                newCtxt.hoistConstant(zeroTensor)
                self.operatorRepresentation['C'] = f'{node.name}_C_Tensor'

            self.operatorRepresentation['size'] = np.prod(newCtxt.lookup(node.inputs[0].name).shape)

        return newCtxt, ret