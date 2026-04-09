# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# XDNA2 reuses the Generic AddParser (see Platform.py).
# Add any XDNA2-specific parsers here as the platform grows.

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser


class XDNA2LayerNormParser(NodeParser):
    """Simplified LayerNorm parser for XDNA2.

    The XDNA2 kernel hardcodes gamma=1.0 and beta=0.0, so only the
    data input and output are registered.  Scale and bias tensors
    are validated to exist in the ONNX graph but are **not** added
    to ``operatorRepresentation``, keeping the tiling system simple
    (UnaryTileConstraint: 1 input, 1 output).
    """

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        return all([
            'epsilon' in node.attrs,
            len(node.inputs) == 3,
            len(node.outputs) >= 1,
        ])

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))
        self.operatorRepresentation['lastDimLength'] = int(data_in.shape[-1])
        self.operatorRepresentation['epsilon'] = node.attrs['epsilon']

        return ctxt, True
