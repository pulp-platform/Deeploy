# ----------------------------------------------------------------------
# File: iLeakyReLUParser.py
#
# SoCDAML Part III - TA reference solution.
# Parser for the iLeakyReLU op. Appended to:
#   Deeploy/Targets/Generic/Parsers.py
# ----------------------------------------------------------------------
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# (in Generic/Parsers.py the following imports already exist:
#  import math; import numpy as np; import onnx_graphsurgeon as gs;
#  from Deeploy.DeeployTypes import NodeParser, NetworkContext)


class iLeakyReLUParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        wellFormed = all([
            len(node.inputs) == 1,
            len(node.outputs) == 1,
            'mul' in node.attrs,
            'shift' in node.attrs,
        ])

        if not wellFormed:
            return False

        mul = int(node.attrs['mul'])
        shift = int(node.attrs['shift'])

        # XPULP has no per-lane multiply for packed int8, so the SIMD kernel
        # can only compute max(x, x >> shift), i.e. mul == 1.
        if mul != 1 or not 0 <= shift < 8:
            return False

        self.operatorRepresentation['mul'] = mul
        self.operatorRepresentation['shift'] = shift
        return True

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node, channels_first: bool = True):
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))
        return ctxt, True
