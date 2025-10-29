# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Targets.Generic.Parsers import GEMMParser, RQGEMMParser


class SnitchGEMMParser(GEMMParser):

    def parseNode(self, node: gs.Node) -> bool:
        ret = super().parseNode(node)

        if not ret:
            return False

        if self.operatorRepresentation['transA']:
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        if not all([
                self.operatorRepresentation['batch'] == 1,
        ]):
            return ctxt, False

        return newCtxt, True


class SnitchRQGEMMParser(RQGEMMParser):

    def parseNode(self, node: gs.Node) -> bool:
        ret = super().parseNode(node)

        if not ret:
            return False

        if self.operatorRepresentation['transA']:
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        if not all([
                self.operatorRepresentation['batch'] == 1,
        ]):
            return ctxt, False

        return newCtxt, True


class iNoNormParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all(['D' in node.attrs, 'mul' in node.attrs, 'n_levels' in node.attrs])

        if ret:
            self.operatorRepresentation.update(node.attrs)
            self.operatorRepresentation['log2D'] = int(math.log2(node.attrs['D']))

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        weights = ctxt.lookup(node.inputs[1].name)
        bias = ctxt.lookup(node.inputs[2].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['weights'] = weights.name
        self.operatorRepresentation['bias'] = bias.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = math.prod(data_in.shape)

        return ctxt, True
