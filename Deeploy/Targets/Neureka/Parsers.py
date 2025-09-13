# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

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
        weight = newCtxt.lookup(self.operatorRepresentation['weight'])

        if not all([
                channels_first == False,
                len(data_in.shape) == 4,
                # LMACAN: weight shape should be equal to 3 because we have to do the neureka's
                #         special weight encoding
                len(weight.shape) == 3,
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

        ch_im_out = node.inputs[1].shape[0]
        ch_im_in = node.inputs[1].shape[1]

        if not all([
                self.operatorRepresentation['kernel_shape'] == [3, 3],
                self.operatorRepresentation['group'] == ch_im_out,
                self.operatorRepresentation['group'] == ch_im_in,
        ]):
            return False

        return True


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

        inputs = ['data_in', 'weight', 'mul', 'add']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

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
