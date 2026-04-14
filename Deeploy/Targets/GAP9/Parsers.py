# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import GEMMParser, RQSParserInterface


class NE16GEMMParser(GEMMParser, RQSParserInterface):
    """Parser for NE16 RequantizedGemm nodes with 5 inputs [A, B, C, mul, scale_n]."""

    def __init__(self):
        super().__init__(noBiasHoisting = True)

    def parseNode(self, node: gs.Node) -> bool:
        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_matmul = GEMMParser.parseNode(self, node)
        ret = all([ret_rqs, ret_matmul, 'shift' in node.attrs, len(node.inputs) == 5])
        if ret:
            self.operatorRepresentation['shift'] = int(node.attrs['shift'].values)
        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = GEMMParser.parseNodeCtxt(self, ctxt, node, channels_first)
        if ret:
            inputs = ['A', 'B', 'C', 'mul', 'scale_n']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            return newCtxt, True
        return ctxt, False
