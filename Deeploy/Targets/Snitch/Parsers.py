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

        if not all([
                self.operatorRepresentation['transA'] == 0,
        ]):
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

        if not all([
                self.operatorRepresentation['transA'] == 0,
        ]):
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
        if not all([
                'D' in node.attrs,
                'mul' in node.attrs,
                'n_levels' in node.attrs,
                len(node.inputs) == 3,
                len(node.outputs) == 1,
        ]):
            return False

        self.operatorRepresentation['log2D'] = int(math.log2(self._unpack_const(node.attrs['D'])))
        self.operatorRepresentation['n_levels'] = int(self._unpack_const(node.attrs['n_levels']))
        self.operatorRepresentation['mul'] = int(self._unpack_const(node.attrs['mul']))
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        for tensor, symName in zip(node.inputs, ["data_in", "weights", "bias"], strict = True):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        for tensor, symName in zip(node.outputs, ["data_out"], strict = True):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        self.operatorRepresentation['size'] = math.prod(ctxt.lookup(node.inputs[0].name).shape)
        return ctxt, True
