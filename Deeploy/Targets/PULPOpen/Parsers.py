# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import Conv2DParser, GEMMParser, RQSConv1DParser, RQSConv2DParser, \
    RQSParserInterface


class PULPConv2DParser(RQSConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['group'] == 1,
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                len(node.inputs) == 4,
                'shift' in node.attrs,
            ])

            self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
            self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
            self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPFPConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['group'] == 1,
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                len(node.inputs) == 2
            ])

            self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
            self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
            self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            return newCtxt, True

        return ctxt, False


class PULPDWConv1DParser(RQSConv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
            ])

            if ret:

                self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
                self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
                self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
                self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][1])
                self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])

                if 'n_levels' in node.attrs:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
                else:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels_out'].values)

                self.operatorRepresentation['signed'] = int(node.attrs['signed'].values)
                self.operatorRepresentation['log2D'] = int(math.log2(node.attrs['div'].values))
            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.operatorRepresentation['group'] == newCtxt.lookup(
                    self.operatorRepresentation['weight']).shape[0]:
                return ctxt, False

            # if not newCtxt.is_global(self.operatorRepresentation['weight']):
            #     return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            # newCtxt.globalObjects[self.operatorRepresentation['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])

            return newCtxt, True

        return ctxt, False


class PULPDWConv2DParser(RQSConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                node.op == 'RequantizedConv',
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
                'shift' in node.attrs,
                any(['n_levels' in node.attrs, 'n_levels_out' in node.attrs]),
                'signed' in node.attrs
            ])

            if ret:
                self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
                self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
                self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
                self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
                self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
                self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
                self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
                self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
                self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
                self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])

                if 'n_levels' in node.attrs:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
                else:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels_out'].values)
                self.operatorRepresentation['signed'] = int(node.attrs['signed'].values)
                self.operatorRepresentation['log2D'] = int(math.log2(node.attrs['div'].values))

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.operatorRepresentation['group'] == newCtxt.lookup(
                    self.operatorRepresentation['weight']).shape[0]:
                return ctxt, False

            data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
            data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
            _ = newCtxt.lookup(self.operatorRepresentation['weight'])

            # if not newCtxt.is_global(self.operatorRepresentation['weight']):
            #     return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            # newCtxt.globalObjects[self.operatorRepresentation['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])

            if channels_first:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_x'] = data_out.shape[2]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[3]
            else:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[3]
                self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]

            return newCtxt, True

        return ctxt, False


class PULPConv1DParser(RQSConv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['group'] == 1,
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
            ])

            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPGEMMParser(GEMMParser, RQSParserInterface):

    def __init__(self):
        super().__init__(noBiasHoisting = True)

    def parseNode(self, node: gs.Node) -> (bool):

        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_matmul = GEMMParser.parseNode(self, node)

        ret = all([
            ret_rqs == True,
            ret_matmul == True,
            'shift' in node.attrs,
            len(node.inputs) == 4,
        ])

        if ret:
            self.operatorRepresentation['shift'] = int(node.attrs['shift'].values)

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['A', 'B', 'C', 'mul']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            return newCtxt, True

        else:
            return ctxt, False


class PULPMatrixVecParser(PULPGEMMParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        M = node.inputs[0].shape[-1 if node.attrs["transA"] else -2]
        batch = math.prod(node.inputs[0].shape[:-2])
        return M == 1 and batch >= 8


class PULPTallGEMMParser(PULPGEMMParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        M = node.inputs[0].shape[-1 if node.attrs["transA"] else -2]
        N = node.inputs[1].shape[-2 if node.attrs["transB"] else -1]
        batch = math.prod(node.inputs[0].shape[:-2])
        return M >= 8 and (M % 8) < (N % 8) and batch < 8
