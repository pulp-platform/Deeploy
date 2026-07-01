# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import Conv2DParser, ConvParser, RQSParserInterface


class NeurekaConv2DBaseParser(Conv2DParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                # No dilation support
                self.operatorRepresentation['dilations'] == [1, 1],
                # Channels have to be last
                'channels_first' in self.operatorRepresentation and not self.operatorRepresentation['channels_first'],
                # Expect "weight_offset" attribute in the node
                "weight_offset" in node.attrs,
        ]):
            return False

        self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
        self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
        self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
        self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
        self.operatorRepresentation['weight_offset'] = int(node.attrs["weight_offset"])

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        # LMACAN: Cannot reuse the Conv2DParser's parserNodeCtxt because it requires the weight shape
        #         to be of length 4 whereas neureka does a specific weight encoding so the shape
        #         ends up being equal to 3.
        newCtxt, ret = ConvParser.parseNodeCtxt(self, ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        # LMACAN: c/p of Conv2DParser's parserNodeCtxt but with a different weight shape check
        #         and enforcing that the channels_first is false
        data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
        data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
        # MARCHIOA: weight depends on the type of convolution so it requires to be parsed by the child parsers
        #           - PW -> 3-dim
        #           - DW -> 4-dim
        #           - Dense -> 4-dim
        # weight = newCtxt.lookup(self.operatorRepresentation['weight'])

        if not all([
                channels_first == False,
                len(data_in.shape) == 4,
                # # LMACAN: weight shape should be equal to 3 because we have to do the neureka's
                # #         special weight encoding
                # len(weight.shape) == 3,
        ]):
            return newCtxt, False

        self.operatorRepresentation['batch'] = data_in.shape[0]
        self.operatorRepresentation['dim_im_in_x'] = data_in.shape[1]
        self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
        self.operatorRepresentation['ch_im_in'] = data_in.shape[3]
        self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
        self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]
        self.operatorRepresentation['ch_im_out'] = data_out.shape[3]

        # No requantization
        self.operatorRepresentation['mul'] = 'NULL'
        self.operatorRepresentation['add'] = 'NULL'
        self.operatorRepresentation['shift'] = 'NULL'

        return newCtxt, True


class NeurekaDWConv2DParser(NeurekaConv2DBaseParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        weights = node.inputs[1]

        # weigths reshaped by the weigths encoder into
        # (cout, cinMajor, bits, weightBandwidthBytes)
        # where:
        # - cout: 1 by definition (it is cin from ONNX)
        # - cinMajor: number of tiles over the channels
        # - bits: weight bit width (only 8 is supported)
        # - weightBandwidthBytes: which is 32 in Siracusa
        if not all([
                self.operatorRepresentation['kernel_shape'] == [3, 3],
                len(weights.shape) == 4,
                weights.shape[0] == 1,  # ch_im_out
        ]):
            return False

        return True

    def parseNodeCtxt(self, ctxt, node, channels_first = True):

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if not ret:
            return False

        weight = newCtxt.lookup(self.operatorRepresentation['weight'])
        if not (len(weight.shape) == 4):
            return False

        return newCtxt, True


class NeurekaRQSDWConv2DParser(NeurekaDWConv2DParser, RQSParserInterface):

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            RQSParserInterface.parseNode(self, node),
            NeurekaDWConv2DParser.parseNode(self, node),
        ])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        inputs = ['data_in', 'weight', 'mul', 'add']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

        return newCtxt, True


class NeurekaPWConv2DParser(NeurekaConv2DBaseParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                self.operatorRepresentation['kernel_shape'] == [1, 1],
                self.operatorRepresentation['group'] == 1,
        ]):
            return False

        return True

    def parseNodeCtxt(self, ctxt, node, channels_first = True):

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if not ret:
            return False

        weight = newCtxt.lookup(self.operatorRepresentation['weight'])
        if not (len(weight.shape) == 3):
            return False

        return newCtxt, True


class NeurekaRQSPWConv2DParser(NeurekaPWConv2DParser, RQSParserInterface):

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            RQSParserInterface.parseNode(self, node),
            NeurekaPWConv2DParser.parseNode(self, node),
        ])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        data_in = ctxt.lookup(node.inputs[0].name)
        weight = ctxt.lookup(node.inputs[1].name)
        mul = ctxt.lookup(node.inputs[2].name)
        add = ctxt.lookup(node.inputs[3].name)

        # The Neureka PW conv's RQS unit only supports per-tensor or
        # per-output-channel requantization: mul/add must have either 1
        # element or one element per output channel (weight's dim 0).
        out_channels = weight.shape[0]
        for tensor in (mul, add):
            size = int(np.prod(tensor.shape))
            if size not in (1, out_channels):
                return ctxt, False

        self.operatorRepresentation['data_in'] = data_in
        self.operatorRepresentation['weight'] = weight
        self.operatorRepresentation['mul'] = mul
        self.operatorRepresentation['add'] = add

        return newCtxt, True


class NeurekaDenseConv2DParser(NeurekaConv2DBaseParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                self.operatorRepresentation['kernel_shape'] == [3, 3],
                self.operatorRepresentation['group'] == 1,
        ]):
            return False

        return True

    def parseNodeCtxt(self, ctxt, node, channels_first = True):

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if not ret:
            return False

        weight = newCtxt.lookup(self.operatorRepresentation['weight'])
        if not (len(weight.shape) == 4):
            return False

        return newCtxt, True


class NeurekaRQSDenseConv2DParser(NeurekaDenseConv2DParser, RQSParserInterface):

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            RQSParserInterface.parseNode(self, node),
            NeurekaDenseConv2DParser.parseNode(self, node),
        ])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        inputs = ['data_in', 'weight', 'mul', 'add']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

        return newCtxt, True
