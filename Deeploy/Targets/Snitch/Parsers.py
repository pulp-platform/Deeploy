# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Targets.Generic.Parsers import AddParser, DivParser, GEMMParser, MulParser, RQGEMMParser


def compute_broadcast_strides(shape1, shape2, out_shape):
    """Compute strides for ONNX/NumPy-style broadcasting.

    Pads both input shapes from the left to match the output ndim,
    then computes strides where broadcast dimensions (size 1) get stride 0.

    Example:
        shape1=[8,8,8], shape2=[8]
        -> strides1=[64,8,1], strides2=[0,0,1]
    """
    ndim = len(out_shape)

    pad1 = [1] * (ndim - len(shape1)) + shape1
    pad2 = [1] * (ndim - len(shape2)) + shape2

    def _calc_strides(padded_shape, out_shape):
        strides = []
        stride = 1
        for i in range(ndim - 1, -1, -1):
            if padded_shape[i] == 1 and out_shape[i] > 1:
                strides.insert(0, 0)
            else:
                strides.insert(0, stride)
            stride *= padded_shape[i] if padded_shape[i] > 1 else 1
        return strides

    strides1 = _calc_strides(pad1, out_shape)
    strides2 = _calc_strides(pad2, out_shape)
    return strides1, strides2


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


class SnitchRMSNormParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        if node.op != 'RMSNorm':
            return False
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            return False
        eps = node.attrs.get('eps', node.attrs.get('epsilon', 1e-6))
        self.operatorRepresentation['eps'] = f"{float(eps):.10e}f"

        stash_type = node.attrs.get('stash_type', 1)
        if stash_type != 1:
            raise ValueError(f"RMSNorm: only stash_type=1 (FP32) is supported, got {stash_type}")

        axis = node.attrs.get('axis', -1)
        self.operatorRepresentation['axis'] = axis

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        weight = ctxt.lookup(node.inputs[1].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['weight'] = weight.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['input_shape'] = list(data_in.shape)
        self.operatorRepresentation['weight_shape'] = list(weight.shape)
        self.operatorRepresentation['output_shape'] = list(data_out.shape)
        self.operatorRepresentation['input_ndim'] = len(data_in.shape)
        self.operatorRepresentation['weight_ndim'] = len(weight.shape)

        input_shape = list(data_in.shape)
        axis = self.operatorRepresentation['axis']
        if axis < 0:
            axis = len(input_shape) + axis

        self.operatorRepresentation['inputSize'] = int(np.prod(input_shape))
        self.operatorRepresentation['NormalizedAxesSize'] = int(np.prod(input_shape[axis:]))
        self.operatorRepresentation['scale'] = node.inputs[1].values

        # Keep old keys for C template compatibility
        self.operatorRepresentation['size'] = int(np.prod(input_shape))
        self.operatorRepresentation['lastDimLength'] = int(input_shape[-1])

        return ctxt, True


class HardSwishParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        """Parse HardSwish node."""

        if node.op != 'HardSwish':
            return False

        # Check basic structure: 1 input and 1 output
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        """Parse HardSwish node with network context."""

        # Get input and output buffers
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        # Store buffer names
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name

        # Calculate size for memory allocation
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))

        return ctxt, True


class SnitchAddParser(AddParser):
    """Inherits from GenericAddParser and adds broadcasting support."""

    def __init__(self):
        super().__init__()

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if not ret:
            return ctxt, False

        shape1 = list(ctxt.lookup(node.inputs[0].name).shape)
        shape2 = list(ctxt.lookup(node.inputs[1].name).shape)
        out_shape = list(ctxt.lookup(node.outputs[0].name).shape)

        self.operatorRepresentation['size'] = int(np.prod(out_shape))
        self.operatorRepresentation['shape1'] = shape1
        self.operatorRepresentation['shape2'] = shape2
        self.operatorRepresentation['out_shape'] = out_shape
        self.operatorRepresentation['ndim'] = len(out_shape)

        need_broadcast = (shape1 != shape2)
        self.operatorRepresentation['need_broadcast'] = need_broadcast

        if need_broadcast:
            strides1, strides2 = compute_broadcast_strides(shape1, shape2, out_shape)
            self.operatorRepresentation['strides1'] = strides1
            self.operatorRepresentation['strides2'] = strides2

        return ctxt, True


class SnitchDivParser(DivParser):
    """Inherits from Generic DivParser and adds broadcasting support."""

    def __init__(self):
        super().__init__()

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if not ret:
            return ctxt, False

        shape1 = list(ctxt.lookup(node.inputs[0].name).shape)
        shape2 = list(ctxt.lookup(node.inputs[1].name).shape)
        out_shape = list(ctxt.lookup(node.outputs[0].name).shape)

        self.operatorRepresentation['shape1'] = shape1
        self.operatorRepresentation['shape2'] = shape2
        self.operatorRepresentation['out_shape'] = out_shape
        self.operatorRepresentation['ndim'] = len(out_shape)
        self.operatorRepresentation['size1'] = int(np.prod(shape1))
        self.operatorRepresentation['size2'] = int(np.prod(shape2))
        self.operatorRepresentation['size'] = int(np.prod(out_shape))

        need_broadcast = (shape1 != shape2)
        self.operatorRepresentation['need_broadcast'] = need_broadcast
        self.operatorRepresentation['is_scalar'] = (np.prod(shape1) == 1 or np.prod(shape2) == 1)

        if need_broadcast:
            strides1, strides2 = compute_broadcast_strides(shape1, shape2, out_shape)
            self.operatorRepresentation['strides1'] = strides1
            self.operatorRepresentation['strides2'] = strides2

        return ctxt, True


class SnitchMulParser(MulParser):
    """Inherits from Generic MulParser and adds broadcasting support."""

    def __init__(self):
        super().__init__()

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if not ret:
            return ctxt, False

        shape1 = list(ctxt.lookup(node.inputs[0].name).shape)
        shape2 = list(ctxt.lookup(node.inputs[1].name).shape)
        out_shape = list(ctxt.lookup(node.outputs[0].name).shape)

        self.operatorRepresentation['shape1'] = shape1
        self.operatorRepresentation['shape2'] = shape2
        self.operatorRepresentation['out_shape'] = out_shape
        self.operatorRepresentation['ndim'] = len(out_shape)
        self.operatorRepresentation['size1'] = int(np.prod(shape1))
        self.operatorRepresentation['size2'] = int(np.prod(shape2))
        self.operatorRepresentation['size'] = int(np.prod(out_shape))

        need_broadcast = (shape1 != shape2)
        self.operatorRepresentation['need_broadcast'] = need_broadcast
        self.operatorRepresentation['is_scalar'] = (np.prod(shape1) == 1 or np.prod(shape2) == 1)

        if need_broadcast:
            strides1, strides2 = compute_broadcast_strides(shape1, shape2, out_shape)
            self.operatorRepresentation['strides1'] = strides1
            self.operatorRepresentation['strides2'] = strides2

        return ctxt, True
