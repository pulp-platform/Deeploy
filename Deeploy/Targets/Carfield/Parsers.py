# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Targets.Generic.Parsers import AddParser, GenericConv2DParser, GenericMaxPool2DParser, Pad2DParser


class CarfieldChannelBroadcastParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        return all([len(node.inputs) == 2, len(node.outputs) == 1])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        if not channels_first:
            return ctxt, False

        data_in_1 = ctxt.lookup(node.inputs[0].name)
        data_in_2 = ctxt.lookup(node.inputs[1].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        dataShape = list(data_in_1.shape)
        operandShape = list(data_in_2.shape)

        if len(dataShape) != 4 or len(operandShape) != 4:
            return ctxt, False

        if operandShape[0] != 1 or operandShape[2] != 1 or operandShape[3] != 1:
            return ctxt, False

        if operandShape[1] != dataShape[1]:
            return ctxt, False

        if list(data_out.shape) != dataShape:
            return ctxt, False

        self.operatorRepresentation['data_in_1'] = data_in_1.name
        self.operatorRepresentation['data_in_2'] = data_in_2.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(dataShape))
        self.operatorRepresentation['channels'] = int(dataShape[1])
        self.operatorRepresentation['spatialSize'] = int(np.prod(dataShape[2:]))

        return ctxt, True


class CarfieldResizeNearestParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            len(node.inputs) == 1,
            len(node.outputs) == 1,
            'scale_h' in node.attrs,
            'scale_w' in node.attrs,
        ])

        if ret:
            self.operatorRepresentation['scale_h'] = int(node.attrs['scale_h'])
            self.operatorRepresentation['scale_w'] = int(node.attrs['scale_w'])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        if not channels_first:
            return ctxt, False

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        inShape = list(data_in.shape)
        outShape = list(data_out.shape)

        if len(inShape) != 4 or len(outShape) != 4:
            return ctxt, False
        if outShape[0] != inShape[0] or outShape[1] != inShape[1]:
            return ctxt, False
        if inShape[0] != 1:
            return ctxt, False
        if outShape[2] != inShape[2] * self.operatorRepresentation['scale_h']:
            return ctxt, False
        if outShape[3] != inShape[3] * self.operatorRepresentation['scale_w']:
            return ctxt, False

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(outShape))
        self.operatorRepresentation['channels'] = int(inShape[1])
        self.operatorRepresentation['H_in'] = int(inShape[2])
        self.operatorRepresentation['W_in'] = int(inShape[3])
        self.operatorRepresentation['H_out'] = int(outShape[2])
        self.operatorRepresentation['W_out'] = int(outShape[3])

        return ctxt, True


class CarfieldConcat2Parser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            'axis' in node.attrs,
            len(node.inputs) == 2,
            len(node.outputs) == 1,
        ])

        if ret and int(node.attrs['axis']) == 1:
            self.operatorRepresentation['axis'] = 1
            return True

        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        if not channels_first:
            return ctxt, False

        data_in_1 = ctxt.lookup(node.inputs[0].name)
        data_in_2 = ctxt.lookup(node.inputs[1].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        shape_1 = list(data_in_1.shape)
        shape_2 = list(data_in_2.shape)
        shape_out = list(data_out.shape)

        if len(shape_1) != 4 or len(shape_2) != 4 or len(shape_out) != 4:
            return ctxt, False
        if shape_1[0] != 1 or shape_2[0] != 1 or shape_out[0] != 1:
            return ctxt, False
        if shape_1[2:] != shape_2[2:] or shape_1[2:] != shape_out[2:]:
            return ctxt, False
        if shape_1[1] + shape_2[1] != shape_out[1]:
            return ctxt, False

        self.operatorRepresentation['data_in_1'] = data_in_1.name
        self.operatorRepresentation['data_in_2'] = data_in_2.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size_1'] = int(np.prod(shape_1))
        self.operatorRepresentation['size_2'] = int(np.prod(shape_2))
        self.operatorRepresentation['size'] = int(np.prod(shape_out))

        return ctxt, True


class CarfieldArgMaxChannelParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            'axis' in node.attrs,
            'keepdims' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1,
        ])

        if not ret:
            return False

        if int(node.attrs['axis']) != 1 or int(node.attrs['keepdims']) != 0:
            return False

        if int(node.attrs.get('select_last_index', 0)) != 0:
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        if not channels_first:
            return ctxt, False

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        inShape = list(data_in.shape)
        outShape = list(data_out.shape)

        if len(inShape) != 4 or len(outShape) != 3:
            return ctxt, False
        if outShape[0] != inShape[0] or outShape[1] != inShape[2] or outShape[2] != inShape[3]:
            return ctxt, False
        if inShape[0] != 1:
            return ctxt, False

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['channels'] = int(inShape[1])
        self.operatorRepresentation['spatialSize'] = int(inShape[2] * inShape[3])
        self.operatorRepresentation['size'] = int(np.prod(outShape))

        return ctxt, True


class CarfieldCastInt32Parser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            'to' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1,
        ])

        return ret and int(node.attrs['to']) == 6

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        if list(data_in.shape) != list(data_out.shape):
            return ctxt, False

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))

        return ctxt, True


# The Generic parsers below accept cases the Carfield kernels do not implement;
# each subclass only adds the restriction, so the rejected node gets no mapping.


class CarfieldAddParser(AddParser):
    """Element-wise Add only: the kernel indexes both inputs with the output index.

    Checked in parseNode because Deeploy broadcasts the shapes before parseNodeCtxt.
    """

    def parseNode(self, node: gs.Node) -> bool:
        shapes = [list(tensor.shape) for tensor in node.inputs + node.outputs]
        return super().parseNode(node) and all(shape == shapes[0] for shape in shapes)


class CarfieldConv2DParser(GenericConv2DParser):
    """Unit stride only: the kernels and the template take no stride."""

    def parseNode(self, node: gs.Node) -> bool:
        return super().parseNode(node) and all(stride == 1 for stride in self.operatorRepresentation['strides'])


class CarfieldMaxPool2DParser(GenericMaxPool2DParser):
    """Batch 1 only: the template issues a single call for one batch."""

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        return newCtxt, ret and self.operatorRepresentation['batch'] == 1


class CarfieldPad2DParser(Pad2DParser):
    """Symmetric vertical padding only: the template advances by 2 * pad_y per channel."""

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False
        pads = self.operatorRepresentation['pads']
        return pads[2] == pads[6]
