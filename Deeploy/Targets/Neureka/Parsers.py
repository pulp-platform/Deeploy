# ----------------------------------------------------------------------
#
# File: Parsers.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import ConvParser, RQSParserInterface


class NeurekaConv2DBaseParser(ConvParser):

    def __init__(self, noBiasHoisting: bool = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                len(node.attrs['pads']) == 4,
                # No dilation support
                self.operatorRepresentation['dilations'] == [1, 1],
                # Channels have to be last
                'channels_first' in self.operatorRepresentation and not self.operatorRepresentation['channels_first'],
                # Expect "weight_offset" attribute in the node
                "weight_offset" in node.attrs,
        ]):
            return False

        self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
        self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
        self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
        self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
        self.operatorRepresentation['padding_x'] = int(self.operatorRepresentation['pads'][0])
        self.operatorRepresentation['padding_y'] = int(self.operatorRepresentation['pads'][1])
        self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
        self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])
        self.operatorRepresentation['bias_shift'] = int(0)
        self.operatorRepresentation['out_shift'] = int(0)
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

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
        data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
        weight = newCtxt.lookup(self.operatorRepresentation['weight'])

        self.operatorRepresentation['batch'] = data_in.shape[0]
        if channels_first:
            self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
            self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
            self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
            self.operatorRepresentation['ch_im_out'] = data_out.shape[1]
            self.operatorRepresentation['dim_im_out_x'] = data_out.shape[2]
            self.operatorRepresentation['dim_im_out_y'] = data_out.shape[3]
        else:
            self.operatorRepresentation['ch_im_in'] = data_in.shape[3]
            self.operatorRepresentation['dim_im_in_x'] = data_in.shape[1]
            self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
            self.operatorRepresentation['ch_im_out'] = data_out.shape[3]
            self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
            self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]

        # No requantization
        self.operatorRepresentation['mul'] = 'NULL'
        self.operatorRepresentation['add'] = 'NULL'
        self.operatorRepresentation['shift'] = 'NULL'

        return newCtxt, True


class NeurekaDWConv2DParser(NeurekaConv2DBaseParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not self.operatorRepresentation['kernel_shape'] == [3, 3]:
            return False
        if self.operatorRepresentation['group'] == 1:
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        data_in = ctxt.lookup(self.operatorRepresentation['data_in'])
        weight = ctxt.lookup(self.operatorRepresentation['weight'])

        if len(data_in.shape) != 4 or len(weight.shape) != 4:
            return ctxt, False

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

        if not self.operatorRepresentation['kernel_shape'] == [1, 1]:
            return False

        # if not self.operatorRepresentation['strides'] == [1, 1]:
        #     return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
        weight = newCtxt.lookup(self.operatorRepresentation['weight'])

        if len(data_in.shape) != 4 or len(weight.shape) != 3:
            return ctxt, False

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

        inputs = ['data_in', 'weight', 'mul', 'add']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

        return newCtxt, True


class NeurekaDenseConv2DParser(NeurekaConv2DBaseParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not self.operatorRepresentation['kernel_shape'] == [3, 3]:
            return False

        if not self.operatorRepresentation['group'] == 1:
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
        weight = newCtxt.lookup(self.operatorRepresentation['weight'])

        if len(data_in.shape) != 4 or len(weight.shape) != 4:
            return ctxt, False

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
