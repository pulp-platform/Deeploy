# ----------------------------------------------------------------------
#
# File: MemPoolParsers.py
#
# Last edited: 13.11.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import MHSAParser


class MemPoolMHSAParser(MHSAParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)
        if wellFormed:
            wellFormed = all([
                self.operatorRepresentation['n_levels'] == 256,
                'preattn_requant_add' in node.attrs,
                'postattn_requant_add' in node.attrs,
                'wo_requant_add' in node.attrs,
                'wq_requant_add' in node.attrs,
                'wk_requant_add' in node.attrs,
                'wv_requant_add' in node.attrs,
            ])

        if wellFormed:
            self.operatorRepresentation['preattn_requant_add'] = node.attrs['preattn_requant_add']
            self.operatorRepresentation['postattn_requant_add'] = node.attrs['postattn_requant_add']
            self.operatorRepresentation['wo_requant_add'] = node.attrs['wo_requant_add']
            self.operatorRepresentation['wq_requant_add'] = node.attrs['wq_requant_add']
            self.operatorRepresentation['wk_requant_add'] = node.attrs['wk_requant_add']
            self.operatorRepresentation['wv_requant_add'] = node.attrs['wv_requant_add']

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        K = ctxt.lookup(self.operatorRepresentation['k'])
        V = ctxt.lookup(self.operatorRepresentation['v'])

        self.operatorRepresentation['E'] = int(K.shape[-1])  # Embedding size

        wellFormed = all([
            K.name == V.name  # K and V has to be the same for ITA
        ])

        return newCtxt, wellFormed


class MemPoolM1HSAParser(MemPoolMHSAParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)
        if wellFormed:
            wellFormed = all([
                self.operatorRepresentation['heads'] == 1,
            ])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret


class MemPoolM2HSAParser(MemPoolMHSAParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)
        if wellFormed:
            wellFormed = all([
                self.operatorRepresentation['heads'] == 2,
            ])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret


class MemPoolITAM4HSAParser(MemPoolMHSAParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)
        if wellFormed:
            wellFormed = all([
                self.operatorRepresentation['heads'] % 4 == 0,
            ])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret
