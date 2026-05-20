# ----------------------------------------------------------------------
# File: iLeakyReLUParser.py  (SoCDAML Part III - Step 2 skeleton)
#
# Paste this class into:
#   Deeploy/Targets/Generic/Parsers.py
#
# Imports already present in that file (math, numpy as np,
# onnx_graphsurgeon as gs, NodeParser, NetworkContext).
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0


class iLeakyReLUParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        # TODO(student): return False if the node doesn't have exactly
        # one input, exactly one output, and both 'mul' and 'shift'
        # attributes. On success, store them into
        # self.operatorRepresentation as ints and return True.
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True):
        # TODO(student): look up the input and output tensors from ctxt
        # using node.inputs[0].name / node.outputs[0].name, and populate
        # self.operatorRepresentation with:
        #     'data_in'  -> input tensor name
        #     'data_out' -> output tensor name
        #     'size'     -> int(np.prod(input_shape))
        # Return (ctxt, True).
        return ctxt, False
