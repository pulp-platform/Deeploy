# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeParser, VariableBuffer


class ConcatParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all(['axis' in node.attrs, len(node.inputs) >= 2, len(node.outputs) == 1])

        if ret:
            self.operatorRepresentation['axis'] = node.attrs['axis']
            return True

        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_out'] = data_out.name

        for idx, _inp in enumerate(node.inputs):
            data_in = ctxt.lookup(_inp.name)
            self.operatorRepresentation[f'data_in_{idx+1}'] = _inp.name

        return ctxt, True


class iRMSNormParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all(['D' in node.attrs, 'n_levels' in node.attrs, len(node.inputs) == 2, len(node.outputs) == 1])

        if ret:

            self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'])
            self.operatorRepresentation['log2D'] = int(math.log2(node.attrs['D']))

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['data_in', 'weight']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        input_shape = list(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['size'] = np.prod(input_shape)
        self.operatorRepresentation['lastDimLength'] = input_shape[-1]
        self.operatorRepresentation['inputSize'] = int(np.prod(input_shape))
        self.operatorRepresentation['NormalizedAxesSize'] = int(input_shape[-1])
        self.operatorRepresentation['scale'] = node.inputs[1].values

        return ctxt, True


class RQSParserInterface():

    def parseNode(self, node: gs.Node) -> bool:
        if not all([
                'div' in node.attrs,
                'n_levels' in node.attrs or 'n_levels_out' in node.attrs,
                'signed' in node.attrs,
        ]):
            return False

        n_levels = node.attrs['n_levels'] if 'n_levels' in node.attrs else node.attrs['n_levels_out']
        self.operatorRepresentation['n_levels'] = int(NodeParser._unpack_const(n_levels))
        self.operatorRepresentation['signed'] = int(NodeParser._unpack_const(node.attrs['signed']))
        self.operatorRepresentation['log2D'] = int(math.log2(NodeParser._unpack_const(node.attrs['div'])))

        return True


class SliceParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        # Scheremo ONNX >= 10
        retNew = all([len(node.inputs) >= 3, len(node.inputs) <= 5, len(node.outputs) == 1])

        # Scheremo ONNX < 10
        retOld = all([len(node.inputs) == 1, 'ends' in node.attrs, 'starts' in node.attrs, len(node.outputs) == 1])

        if not (retNew or retOld):
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in_shape'] = data_in.shape
        self.operatorRepresentation['data_out_shape'] = data_out.shape
        self.operatorRepresentation['dims'] = len(data_in.shape)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name

        if len(node.inputs) <= 1:
            values = node.attrs['starts']
            startsTensor = gs.Constant(f'{node.name}_Starts_Tensor', values = values)
            ctxt.hoistConstant(startsTensor)
            node.inputs.append(startsTensor)
        if len(node.inputs) <= 2:
            values = node.attrs['ends']
            endsTensor = gs.Constant(f'{node.name}_Ends_Tensor', values = values)
            ctxt.hoistConstant(endsTensor)
            node.inputs.append(endsTensor)
        if len(node.inputs) <= 3:
            values = np.array(list(range(self.operatorRepresentation['dims'])))
            axesTensor = gs.Constant(f'{node.name}_Axes_Tensor', values = values)
            ctxt.hoistConstant(axesTensor)
            node.inputs.append(axesTensor)
        if len(node.inputs) <= 4:
            values = np.ones((self.operatorRepresentation['dims']))
            stepsTensor = gs.Constant(f'{node.name}_Steps_Tensor', values = values)
            ctxt.hoistConstant(stepsTensor)
            node.inputs.append(stepsTensor)

        self.operatorRepresentation['starts'] = node.inputs[1].name
        self.operatorRepresentation['ends'] = node.inputs[2].name

        self.operatorRepresentation['axes'] = node.inputs[3].name
        self.operatorRepresentation['steps'] = node.inputs[4].name

        return ctxt, True


class TransposeParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all(['perm' in node.attrs, len(node.inputs) == 1, len(node.outputs) == 1])

        if ret:
            self.operatorRepresentation['perm'] = node.attrs['perm']
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in_shape'] = data_in.shape
        self.operatorRepresentation['data_out_shape'] = data_out.shape
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['data_in_size'] = np.prod(data_in.shape)
        self.operatorRepresentation['data_out_size'] = np.prod(data_out.shape)

        return ctxt, True


class MaxPoolParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([
            'ceil_mode' in node.attrs, 'kernel_shape' in node.attrs, 'pads' in node.attrs, 'strides' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) >= 1
        ])

        if ret:
            self.operatorRepresentation['ceil_mode'] = node.attrs['ceil_mode']
            self.operatorRepresentation['pads'] = node.attrs['pads']
            self.operatorRepresentation['kernel_shape'] = node.attrs['kernel_shape']
            self.operatorRepresentation['strides'] = node.attrs['strides']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['data_in_size'] = np.prod(data_in.shape)
        self.operatorRepresentation['data_out_size'] = np.prod(data_out.shape)

        return ctxt, True


class MaxPool1DParser(MaxPoolParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = super().parseNode(node)
        wellFormed = False
        if ret:
            pads = self.operatorRepresentation['pads']
            kernel_shape = self.operatorRepresentation['kernel_shape']
            strides = self.operatorRepresentation['strides']
            # 1D: pads should be length 2, kernel_shape length 1, strides length 1
            if len(pads) == 2 and len(kernel_shape) == 1 and len(strides) == 1:
                wellFormed = True
                self.operatorRepresentation['padding_y'] = int(pads[0])
                self.operatorRepresentation['padding_y_right'] = int(pads[1])
                self.operatorRepresentation['stride_y'] = int(strides[0])
                self.operatorRepresentation['dim_kernel_y'] = int(kernel_shape[0])
        return wellFormed

    def parseNodeCtxt(self, ctxt, node, channels_first = True):
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
            data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
            self.operatorRepresentation['batch'] = data_in.shape[0]
            if channels_first:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]
            else:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[1]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[2]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[1]
            if len(data_in.shape) == 3 and len(data_out.shape) == 3:
                return newCtxt, True
        return ctxt, False


class MaxPool2DParser(MaxPoolParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = super().parseNode(node)
        wellFormed = False
        if ret:
            pads = self.operatorRepresentation['pads']
            kernel_shape = self.operatorRepresentation['kernel_shape']
            strides = self.operatorRepresentation['strides']
            if len(pads) == 4 and len(kernel_shape) == 2 and len(strides) == 2:
                wellFormed = True

            self.operatorRepresentation['padding_x'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_y'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][3])
            self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])
            self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        wellFormed = False
        if ret:
            data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
            data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])

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

            if len(data_in.shape) == 4 and len(data_out.shape) == 4:
                wellFormed = True

        return newCtxt, wellFormed


class PadParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([
            'mode' in node.attrs, 'pads' in node.attrs, 'value' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.operatorRepresentation['mode'] = node.attrs['mode']

            try:
                self.operatorRepresentation['pads'] = [int(p) for p in node.attrs['pads']]
            except Exception as e:
                self.operatorRepresentation['pads'] = node.attrs['pads']

            self.operatorRepresentation['value'] = node.attrs['value']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['data_in_size'] = np.prod(data_in.shape)
        self.operatorRepresentation['data_out_size'] = np.prod(data_out.shape)

        return ctxt, True


class Pad2DParser(PadParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = super().parseNode(node)
        wellFormed = False
        if ret:
            pads = self.operatorRepresentation['pads']
            if len(pads) == 8 and pads[0] == 0 and pads[4] == 0 \
            and pads[1] == 0 and pads[5] == 0:
                wellFormed = True
                self.operatorRepresentation['pad_x'] = int(pads[3])
                self.operatorRepresentation['pad_y'] = int(pads[2])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        wellFormed = False
        if ret:
            data_in = newCtxt.lookup(node.inputs[0].name)
            data_out = newCtxt.lookup(node.outputs[0].name)
            if len(data_in.shape) == 4:
                wellFormed = True
                self.operatorRepresentation['batch'] = data_in.shape[0]
                if channels_first:
                    self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
                    self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
                    self.operatorRepresentation['dim_im_in_ch'] = data_in.shape[1]
                    self.operatorRepresentation['dim_im_out_x'] = data_out.shape[2]
                    self.operatorRepresentation['dim_im_out_y'] = data_out.shape[3]
                    self.operatorRepresentation['dim_im_out_ch'] = data_out.shape[1]
                else:
                    self.operatorRepresentation['dim_im_in_x'] = data_in.shape[1]
                    self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
                    self.operatorRepresentation['dim_im_in_ch'] = data_in.shape[3]
                    self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
                    self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]
                    self.operatorRepresentation['dim_im_out_ch'] = data_out.shape[3]
        return newCtxt, wellFormed


class Pad1DParser(PadParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = super().parseNode(node)
        wellFormed = False
        if ret:
            pads = self.operatorRepresentation['pads']
            if len(pads) == 6 and pads[0] == 0 and pads[3] == 0 \
            and pads[1] == 0 and pads[4] == 0:
                wellFormed = True
                self.operatorRepresentation['pad_y'] = int(pads[2])
                self.operatorRepresentation['pad_x'] = 0

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        wellFormed = False
        if ret:
            data_in = newCtxt.lookup(node.inputs[0].name)
            data_out = newCtxt.lookup(node.outputs[0].name)
            if len(data_in.shape) == 3:
                wellFormed = True
                self.operatorRepresentation['batch'] = data_in.shape[0]
                self.operatorRepresentation['dim_im_in_x'] = 1
                self.operatorRepresentation['dim_im_out_x'] = 1
                if channels_first:
                    self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
                    self.operatorRepresentation['dim_im_in_ch'] = data_in.shape[1]
                    self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]
                    self.operatorRepresentation['dim_im_out_ch'] = data_out.shape[1]
                else:
                    self.operatorRepresentation['dim_im_in_y'] = data_in.shape[1]
                    self.operatorRepresentation['dim_im_in_ch'] = data_in.shape[2]
                    self.operatorRepresentation['dim_im_out_y'] = data_out.shape[1]
                    self.operatorRepresentation['dim_im_out_ch'] = data_out.shape[2]
        return newCtxt, wellFormed


class AddParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([len(node.inputs) == 2, len(node.outputs) == 1])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in_1 = ctxt.lookup(node.inputs[0].name)
        data_in_2 = ctxt.lookup(node.inputs[1].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in_1'] = data_in_1.name
        self.operatorRepresentation['data_in_2'] = data_in_2.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_out.shape)

        # Check if broadcasting is needed
        shape1 = list(data_in_1.shape)
        shape2 = list(data_in_2.shape)
        out_shape = list(data_out.shape)

        need_broadcast = (shape1 != out_shape) or (shape2 != out_shape)
        self.operatorRepresentation['need_broadcast'] = need_broadcast

        if need_broadcast:
            # Calculate strides for broadcasting
            ndim = len(out_shape)

            # Compute strides for input 1
            strides1 = [1] * ndim
            for i in range(ndim - 1, -1, -1):
                if i < len(shape1) and shape1[i] == out_shape[i]:
                    if i == ndim - 1:
                        strides1[i] = 1
                    else:
                        strides1[i] = strides1[i + 1] * shape1[i + 1] if (
                            i + 1 < len(shape1) and shape1[i + 1] == out_shape[i + 1]) else strides1[i + 1]
                else:
                    strides1[i] = 0  # Broadcast dimension

            # Compute strides for input 2
            strides2 = [1] * ndim
            for i in range(ndim - 1, -1, -1):
                if i < len(shape2) and shape2[i] == out_shape[i]:
                    if i == ndim - 1:
                        strides2[i] = 1
                    else:
                        strides2[i] = strides2[i + 1] * shape2[i + 1] if (
                            i + 1 < len(shape2) and shape2[i + 1] == out_shape[i + 1]) else strides2[i + 1]
                else:
                    strides2[i] = 0  # Broadcast dimension

            self.operatorRepresentation['ndim'] = ndim
            self.operatorRepresentation['strides1'] = strides1
            self.operatorRepresentation['strides2'] = strides2
            self.operatorRepresentation['out_shape'] = out_shape

        return ctxt, True


class ReduceParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all(['axes' in node.attrs, 'keepdims' in node.attrs, len(node.inputs) == 1, len(node.outputs) == 1])

        if ret:
            if isinstance(node.attrs['axes'], int):
                self.operatorRepresentation['axes'] = [node.attrs['axes']]
            else:
                self.operatorRepresentation['axes'] = node.attrs['axes']
            self.operatorRepresentation['keepdims'] = int(node.attrs['keepdims'])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['data_in_shape'] = data_in.shape
        self.operatorRepresentation['data_out_shape'] = data_out.shape
        self.operatorRepresentation['size'] = np.prod(data_in.shape)
        self.operatorRepresentation['axisLength'] = data_in.shape[self.operatorRepresentation['axes'][0]]

        return ctxt, True


class ReduceMeanParser(ReduceParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        if 1 <= len(node.inputs) and ("axes" not in node.attrs):
            # Float node, requiring 1 or 2 inputs (ONNX opset version >= 18).
            # "axes" input is optional.
            # If axes is not provided, then reduction will happen over all dimensions.
            #
            # WARNING: noop_with_empty_axes attribute not handled

            wellFormed = all(['keepdims' in node.attrs, 1 <= len(node.inputs) <= 2, len(node.outputs) == 1])

            if wellFormed:
                self.operatorRepresentation['keepdims'] = int(node.attrs['keepdims'])
        else:
            # Integer node, requiring 1 input (ONNX opset version < 18)
            wellFormed = super().parseNode(node)

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        if 1 <= len(node.inputs) and ("axes" not in node.attrs):
            # Extract context information for Float ReduceMean node (ONNX opset version >= 18)
            data_in = ctxt.lookup(node.inputs[0].name)
            data_out = ctxt.lookup(node.outputs[0].name)

            # Extract axes as numpy sorted array
            # If not provided, according to ONNX specification, reduction will happen over all dimensions
            if len(node.inputs) == 2:
                axes = ctxt.lookup(node.inputs[1].name)

                # Mark the axes variable to be excluded from the context, since only used in the template, as part of the operator representation
                axes._live = False
                axes._deploy = False

                # Sort axes
                axes = axes.values
                axes.sort()
            else:
                axes = np.array(list(range(len(data_in.shape))))

            # Remove axes reduced over singleton dimensions
            # Keep first axis if only singleton dimensions are reduced
            nonSingletonAxes = []
            for axis in axes:
                if data_in.shape[axis] != 1:
                    nonSingletonAxes.append(axis)
            if len(nonSingletonAxes) == 0:
                nonSingletonAxes.append(axes[0])
            axes = np.array(nonSingletonAxes)

            # Update operator representation
            self.operatorRepresentation['data_in'] = data_in.name
            self.operatorRepresentation['data_out'] = data_out.name

            self.operatorRepresentation['data_in_shape'] = data_in.shape
            self.operatorRepresentation['data_out_shape'] = data_out.shape

            self.operatorRepresentation['size'] = np.prod(data_in.shape)

            self.operatorRepresentation['axes'] = axes
            self.operatorRepresentation['axisLength'] = data_in.shape[axes[0]]

            return ctxt, True
        else:
            newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
            return newCtxt, ret


class ReduceSumParser(ReduceParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        return newCtxt, ret


class SoftmaxParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) == 1, len(node.outputs) == 1])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)
        if 'axis' in node.attrs:
            self.operatorRepresentation['axis'] = int(node.attrs['axis'])
            axis = self.operatorRepresentation['axis']
            self.operatorRepresentation['lastDimLength'] = data_in.shape[axis]
        else:
            self.operatorRepresentation['lastDimLength'] = data_in.shape[-1]

        return ctxt, True


class SoftmaxGradParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) == 2, len(node.outputs) == 1])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        upstream_grad = ctxt.lookup(node.inputs[0].name)
        softmax_output = ctxt.lookup(node.inputs[1].name)
        softmax_grad = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['upstream_grad'] = upstream_grad.name
        self.operatorRepresentation['softmax_output'] = softmax_output.name
        self.operatorRepresentation['softmax_grad'] = softmax_grad.name
        self.operatorRepresentation['size'] = np.prod(upstream_grad.shape)
        if 'axis' in node.attrs:
            self.operatorRepresentation['axis'] = int(node.attrs['axis'])
            axis = self.operatorRepresentation['axis']
            self.operatorRepresentation['lastDimLength'] = upstream_grad.shape[axis]
        else:
            self.operatorRepresentation['lastDimLength'] = upstream_grad.shape[-1]
        return ctxt, True


class iSoftmaxParser(SoftmaxParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        wellFormed = super().parseNode(node)

        if wellFormed:
            wellFormed = all([
                'coeffA' in node.attrs,
                'coeffB' in node.attrs,
                'coeffC' in node.attrs,
                'log2' in node.attrs,
            ])

        if wellFormed:
            self.operatorRepresentation['coeffA'] = int(node.attrs['coeffA'].values)
            self.operatorRepresentation['coeffB'] = int(node.attrs['coeffB'].values)
            self.operatorRepresentation['coeffC'] = int(node.attrs['coeffC'].values)
            self.operatorRepresentation['log2'] = int(node.attrs['log2'].values)
            self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret


class ITAMaxParser(SoftmaxParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        wellFormed = super().parseNode(node)

        ret = all(['n_levels' in node.attrs])

        if ret and wellFormed:
            self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
            return True

        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret


class ITAPartialMaxParser(SoftmaxParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        wellFormed = super().parseNode(node)

        ret = all(['group_width' in node.attrs, 'n_levels' in node.attrs])

        if ret and wellFormed:
            self.operatorRepresentation['group_width'] = int(node.attrs['group_width'])
            self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
            return True

        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret


class GELUParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) >= 1, len(node.outputs) == 1])

        if 'b' in node.attrs and 'one' in node.attrs:
            self.operatorRepresentation['b'] = node.attrs['b']
            self.operatorRepresentation['one'] = node.attrs['one']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)

        return ctxt, True


class GELUGradParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) == 2, len(node.outputs) == 1])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        upstream_grad = ctxt.lookup(node.inputs[0].name)
        gelu_input = ctxt.lookup(node.inputs[1].name)
        gelu_grad = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['grad_in'] = upstream_grad.name
        self.operatorRepresentation['data_in'] = gelu_input.name
        self.operatorRepresentation['grad_out'] = gelu_grad.name
        self.operatorRepresentation['size'] = np.prod(upstream_grad.shape)

        return ctxt, True


class RQSiGELUParser(GELUParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = all([
            len(node.inputs) == 4,
        ])

        ret = super().parseNode(node)
        ret_RQ = all(['b' in node.attrs, 'one' in node.attrs])

        return (ret and ret_RQ and wellFormed)

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            inputs = ['data_in', 'mul', 'add', 'shift']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            return newCtxt, True
        return ctxt, False


class iHardswishParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all(['one_over_six' in node.attrs, 'six' in node.attrs, 'three' in node.attrs])

        if ret:
            self.operatorRepresentation['one_over_six'] = node.attrs['one_over_six']
            self.operatorRepresentation['six'] = node.attrs['six']
            self.operatorRepresentation['three'] = node.attrs['three']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)

        return ctxt, True


class iNoNormParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all(['D' in node.attrs, 'mul' in node.attrs, 'n_levels' in node.attrs])

        if ret:
            self.operatorRepresentation['D'] = node.attrs['D']
            self.operatorRepresentation['log2D'] = int(np.log2(node.attrs['D'].values).tolist()[0])
            self.operatorRepresentation['mul'] = int(node.attrs['mul'].values.tolist()[0])
            self.operatorRepresentation['n_levels'] = node.attrs['n_levels']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        weights = ctxt.lookup(node.inputs[1].name)
        bias = ctxt.lookup(node.inputs[2].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['weights'] = weights.name
        self.operatorRepresentation['bias'] = bias.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)

        return ctxt, True


class RQSiHardswishParser(iHardswishParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node):

        wellFormed = all([len(node.inputs) == 1, 'mul' in node.attrs, 'add' in node.attrs, 'shift' in node.attrs])
        ret = super().parseNode(node)

        if ret and wellFormed:
            self.operatorRepresentation['mul'] = node.attrs['mul']
            self.operatorRepresentation['add'] = node.attrs['add']
            self.operatorRepresentation['shift'] = node.attrs['shift']

            return True

        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            inputs = ['data_in']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            return newCtxt, True
        return ctxt, False


class GatherParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        if not (len(node.inputs) == 2 and len(node.outputs) == 1):
            return False

        indices_shape = node.inputs[1].shape
        assert np.prod(indices_shape) == 1, f"Only indices of size 1 supported. Got indices of shape {indices_shape}"

        self.operatorRepresentation['axis'] = node.attrs['axis'] if 'axis' in node.attrs else 0
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['data_in', 'indices']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        axis = self.operatorRepresentation['axis']
        shape = ctxt.lookup(node.inputs[0].name).shape
        self.operatorRepresentation['batch'] = np.prod(shape[:axis])
        self.operatorRepresentation['batch_length'] = np.prod(shape[axis:])
        self.operatorRepresentation['axis_length'] = np.prod(shape[axis + 1:])
        self.operatorRepresentation['index'] = int(node.inputs[1].values.item())

        return ctxt, True


class FlattenParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all(['axis' in node.attrs, len(node.inputs) == 1, len(node.outputs) == 1])

        if ret:
            self.operatorRepresentation['axis'] = node.attrs['axis']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['data_in']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        return ctxt, True


class UnsqueezeParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        # ONNX v11: 'axes' is a node attribute
        if 'axes' in node.attrs:
            ret = all(['axes' in node.attrs, len(node.inputs) == 1, len(node.outputs) == 1])
        # ONNX v13+: 'axes' becomes an input with the data
        # Source: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
        else:
            ret = all([len(node.inputs) == 2, len(node.outputs) == 1])

        if ret and 'axes' in node.attrs:
            axes_attr = node.attrs['axes']
            self.operatorRepresentation['axes'] = [int(axes_attr)] if isinstance(axes_attr, int) \
                else [int(a) for a in axes_attr]
        # For opset 13+, axes will be extracted from the second input in parseNodeCtxt

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        outputs = ['data_out']
        if len(node.inputs) == 1:
            inputs = ['data_in']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name
        else:
            data_in = ctxt.lookup(node.inputs[0].name)
            data_out = ctxt.lookup(node.outputs[0].name)
            self.operatorRepresentation['data_in'] = data_in.name
            self.operatorRepresentation['data_out'] = data_out.name
            # axes must be a constant; extract values
            axes_buf = ctxt.lookup(node.inputs[1].name)
            assert hasattr(axes_buf, 'values'), "Unsqueeze: expected constant 'axes' input for opset 13+"
            axes_vals = np.array(axes_buf.values).astype(int).flatten().tolist()
            self.operatorRepresentation['axes'] = axes_vals
            # Do not deploy the axes tensor
            axes_buf._live = False
            axes_buf._deploy = False

        return ctxt, True


class ReluParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([len(node.inputs) == 1, len(node.outputs) == 1])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)

        return ctxt, True


class ReshapeParser(NodeParser):

    def parseNode(self, node: gs.Node) -> (bool):
        ret = all([len(node.inputs) == 2, len(node.outputs) == 1])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        for tensor, symName in zip(node.inputs, ['data_in', 'shape']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        for tensor, symName in zip(node.outputs, ['data_out']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        return ctxt, True


class RequantShiftParser(NodeParser, RQSParserInterface):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        if not RQSParserInterface.parseNode(self, node):
            return False
        return len(node.inputs) == 3 and len(node.outputs) == 1

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        _ = channels_first

        inputs = ['data_in', 'mul', 'add']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        data_in = ctxt.lookup(node.inputs[0].name)
        assert isinstance(data_in, VariableBuffer)
        shape = data_in.shape

        assert len(shape) >= 2, f"Unsupported shape length ({len(shape)}). Supported shape lengths greater then 2"

        # Assumes shape [ Batch, Channels, ...]
        self.operatorRepresentation['batch'] = shape[0]
        self.operatorRepresentation['channels'] = shape[1]
        self.operatorRepresentation['channel_width'] = np.prod(shape[2:]) if len(shape) > 2 else 1
        self.operatorRepresentation['size'] = np.prod(shape)

        return ctxt, True


class UniformRequantShiftParser(RequantShiftParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        ret1 = super().parseNode(node)

        ret2 = all([
            np.prod(node.inputs[1].values.shape) == 1,
            np.prod(node.inputs[2].values.shape) == 1,
        ])

        return (ret1 and ret2)


class MulParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = all([
            len(node.inputs) == 2,
            len(node.outputs) == 1,
        ])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['A', 'B']
        outputs = ['C']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['sizeB'] = np.prod(ctxt.lookup(node.inputs[1].name).shape)

        return ctxt, True


class ConvParser(NodeParser):

    def __init__(self, noBiasHoisting):
        super().__init__()
        self.noBiasHoisting = noBiasHoisting

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = all([
            'dilations' in node.attrs,
            'group' in node.attrs,
            'pads' in node.attrs,
            'strides' in node.attrs,
            len(node.outputs) == 1,
        ])

        if wellFormed:
            self.operatorRepresentation['group'] = node.attrs['group']
            self.operatorRepresentation['pads'] = node.attrs['pads']
            self.operatorRepresentation['strides'] = node.attrs['strides']
            self.operatorRepresentation['dilations'] = node.attrs['dilations']

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['data_in', 'weight']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            if idx < len(inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        if len(node.inputs) == 3:
            self.operatorRepresentation['bias'] = ctxt.lookup(node.inputs[2].name).name
        else:
            if not self.noBiasHoisting:
                values = np.zeros((1))
                zeroTensor = gs.Constant(f'{node.name}_Bias_Tensor', values = values)
                ctxt.hoistConstant(zeroTensor)
                node.inputs.append(zeroTensor)
                self.operatorRepresentation['bias'] = f'{node.name}_Bias_Tensor'

        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)

        return ctxt, True


class Conv2DParser(ConvParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        ret = False

        if wellFormed:
            ret = all([
                # Make sure strides are 2D
                len(node.attrs['strides']) == 2,
                len(node.attrs['pads']) == 4,
                len(node.attrs['dilations']) == 2,
            ])

        if ret:
            if 'kernel_shape' not in node.attrs:
                node.attrs['kernel_shape'] = node.inputs[1].shape[-2:]
            self.operatorRepresentation['kernel_shape'] = node.attrs['kernel_shape']
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

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
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

            if len(data_in.shape) == 4 and len(weight.shape) == 4:
                return newCtxt, True

        return ctxt, False


class RQSConv2DParser(Conv2DParser, RQSParserInterface):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_conv = Conv2DParser.parseNode(self, node)

        ret = all([
            ret_rqs == True,
            ret_conv == True,
        ])

        return ret


class Conv1DParser(ConvParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        ret = False

        if wellFormed:
            ret = all([
                # Make sure strides are 2D
                len(node.attrs['strides']) == 1,
                len(node.attrs['pads']) == 2,
                len(node.attrs['dilations']) == 1,
            ])

        if ret:
            if 'kernel_shape' not in node.attrs:
                node.attrs['kernel_shape'] = node.inputs[1].shape[-1:]
            self.operatorRepresentation['kernel_shape'] = node.attrs['kernel_shape']
            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['padding_y'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])
            self.operatorRepresentation['bias_shift'] = int(0)
            self.operatorRepresentation['out_shift'] = int(0)

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
            data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
            weight = newCtxt.lookup(self.operatorRepresentation['weight'])

            self.operatorRepresentation['batch'] = data_in.shape[0]
            self.operatorRepresentation['dim_im_in_x'] = 1

            # Necessary, since we use the same Convlayer for all convolutions
            self.operatorRepresentation['dim_im_out_x'] = 1

            if channels_first:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]
            else:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[1]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[2]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[1]

            self.operatorRepresentation[
                'batchOffsetIn'] = self.operatorRepresentation['ch_im_in'] * self.operatorRepresentation['dim_im_in_y']
            self.operatorRepresentation['batchOffsetOut'] = self.operatorRepresentation[
                'ch_im_out'] * self.operatorRepresentation['dim_im_out_y']

            if len(data_in.shape) == 3 and len(weight.shape) == 3:
                return newCtxt, True

        return ctxt, False


class RQSConv1DParser(Conv1DParser, RQSParserInterface):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_conv = Conv1DParser.parseNode(self, node)

        ret = all([
            ret_rqs == True,
            ret_conv == True,
        ])

        return ret


class MHSAParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([
            'preattn_requant_mul' in node.attrs, 'preattn_requant_div' in node.attrs, 'postattn_requant_mul'
            in node.attrs, 'postattn_requant_div' in node.attrs, 'wo_requant_mul' in node.attrs, 'wo_requant_div'
            in node.attrs, 'wq_requant_mul' in node.attrs, 'wq_requant_div' in node.attrs, 'wk_requant_mul'
            in node.attrs, 'wk_requant_div' in node.attrs, 'wv_requant_mul' in node.attrs, 'wv_requant_div'
            in node.attrs, 'n_levels' in node.attrs, 'dim' in node.attrs, 'dim_head' in node.attrs, 'heads'
            in node.attrs, 'signed' in node.attrs,
            len(node.inputs) == 11,
            len(node.outputs) == 1
        ])

        if ret:
            self.operatorRepresentation['preattn_requant_mul'] = node.attrs['preattn_requant_mul']
            self.operatorRepresentation['preattn_requant_div'] = node.attrs['preattn_requant_div']
            self.operatorRepresentation['postattn_requant_mul'] = node.attrs['postattn_requant_mul']
            self.operatorRepresentation['postattn_requant_div'] = node.attrs['postattn_requant_div']
            self.operatorRepresentation['wo_requant_mul'] = node.attrs['wo_requant_mul']
            self.operatorRepresentation['wo_requant_div'] = node.attrs['wo_requant_div']
            self.operatorRepresentation['wq_requant_mul'] = node.attrs['wq_requant_mul']
            self.operatorRepresentation['wq_requant_div'] = node.attrs['wq_requant_div']
            self.operatorRepresentation['wk_requant_mul'] = node.attrs['wk_requant_mul']
            self.operatorRepresentation['wk_requant_div'] = node.attrs['wk_requant_div']
            self.operatorRepresentation['wv_requant_mul'] = node.attrs['wv_requant_mul']
            self.operatorRepresentation['wv_requant_div'] = node.attrs['wv_requant_div']
            self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'])
            self.operatorRepresentation['dim'] = int(node.attrs['dim'])  # Sequence Length
            self.operatorRepresentation['dim_head'] = int(node.attrs['dim_head'])  # Projection Size
            self.operatorRepresentation['heads'] = int(node.attrs['heads'])
            self.operatorRepresentation['signed'] = int(node.attrs['signed'])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = [
            'q', 'k', 'v', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias', 'wv_weight', 'wv_bias', 'wo_weight',
            'wo_bias'
        ]
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
            self.operatorRepresentation[inputs[idx] + '_shape'] = ctxt.lookup(inputNode.name).shape
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name
            self.operatorRepresentation[outputs[idx] + '_shape'] = ctxt.lookup(outputNode.name).shape

        self.operatorRepresentation['size'] = np.sum([np.prod(ctxt.lookup(x.name).shape) for x in node.inputs])
        # self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)

        return ctxt, True


class LinearAttentionParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([
            'preattn_requant_mul' in node.attrs, 'preattn_requant_div' in node.attrs, 'normalizer_requant_mul'
            in node.attrs, 'normalizer_requant_div' in node.attrs, 'postattn_requant_mul' in node.attrs,
            'postattn_requant_div' in node.attrs, 'wo_requant_mul' in node.attrs, 'wo_requant_div' in node.attrs,
            'wq_requant_mul' in node.attrs, 'wq_requant_div' in node.attrs, 'wk_requant_mul' in node.attrs,
            'wk_requant_div' in node.attrs, 'wv_requant_mul' in node.attrs, 'wv_requant_div' in node.attrs, 'Delta'
            in node.attrs, 'eps' in node.attrs, 'act_type' in node.attrs, 'n_levels' in node.attrs, 'dim' in node.attrs,
            'dim_head' in node.attrs, 'heads' in node.attrs,
            len(node.inputs) == 11,
            len(node.outputs) == 1
        ])

        if ret:
            self.operatorRepresentation['preattn_requant_mul'] = int(node.attrs['preattn_requant_mul'].values)
            self.operatorRepresentation['preattn_requant_shift'] = int(node.attrs['preattn_requant_shift'].values)
            self.operatorRepresentation['preattn_requant_div'] = int(
                math.log2(int(node.attrs['preattn_requant_div'].values)))
            self.operatorRepresentation['normalizer_requant_mul'] = int(node.attrs['normalizer_requant_mul'].values)
            self.operatorRepresentation['normalizer_requant_shift'] = int(node.attrs['normalizer_requant_shift'].values)
            self.operatorRepresentation['normalizer_requant_div'] = int(
                math.log2(int(node.attrs['normalizer_requant_div'].values)))
            self.operatorRepresentation['postattn_requant_mul'] = int(node.attrs['postattn_requant_mul'].values)
            self.operatorRepresentation['postattn_requant_shift'] = int(node.attrs['postattn_requant_shift'].values)
            self.operatorRepresentation['postattn_requant_div'] = int(
                math.log2(int(node.attrs['postattn_requant_div'].values)))
            self.operatorRepresentation['wo_requant_mul'] = int(node.attrs['wo_requant_mul'].values)
            self.operatorRepresentation['wo_requant_shift'] = int(node.attrs['wo_requant_shift'].values)
            self.operatorRepresentation['wo_requant_div'] = int(math.log2(int(node.attrs['wo_requant_div'].values)))
            self.operatorRepresentation['wq_requant_mul'] = int(node.attrs['wq_requant_mul'].values)
            self.operatorRepresentation['wq_requant_shift'] = int(node.attrs['wq_requant_shift'].values)
            self.operatorRepresentation['wq_requant_div'] = int(math.log2(int(node.attrs['wq_requant_div'].values)))
            self.operatorRepresentation['wk_requant_mul'] = int(node.attrs['wk_requant_mul'].values)
            self.operatorRepresentation['wk_requant_shift'] = int(node.attrs['wk_requant_shift'].values)
            self.operatorRepresentation['wk_requant_div'] = int(math.log2(int(node.attrs['wk_requant_div'].values)))
            self.operatorRepresentation['wv_requant_mul'] = int(node.attrs['wv_requant_mul'].values)
            self.operatorRepresentation['wv_requant_shift'] = int(node.attrs['wv_requant_shift'].values)
            self.operatorRepresentation['wv_requant_div'] = int(math.log2(int(node.attrs['wv_requant_div'].values)))
            self.operatorRepresentation['Delta'] = int(node.attrs['Delta'])
            self.operatorRepresentation['eps'] = int(node.attrs['eps'])
            self.operatorRepresentation['act_type'] = int(node.attrs['act_type'])
            self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
            self.operatorRepresentation['dim'] = int(node.attrs['dim'].values)
            self.operatorRepresentation['dim_head'] = int(node.attrs['dim_head'].values)
            self.operatorRepresentation['heads'] = int(node.attrs['heads'].values)

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = [
            'q', 'k', 'v', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias', 'wv_weight', 'wv_bias', 'wo_weight',
            'wo_bias'
        ]
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['q_shape'] = ctxt.lookup(node.inputs[0].name).shape

        return ctxt, True


class CLCAParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([
            'Delta' in node.attrs, 'eps' in node.attrs, 'eta' in node.attrs, 'act_type' in node.attrs, 'n_levels'
            in node.attrs, 'dim' in node.attrs, 'dim_head' in node.attrs, 'out_dim' in node.attrs, 'heads'
            in node.attrs,
            len(node.inputs) == 29,
            len(node.outputs) == 1
        ])

        if ret:
            self.operatorRepresentation['Delta'] = int(node.attrs['Delta'])
            self.operatorRepresentation['eps'] = int(node.attrs['eps'])
            self.operatorRepresentation['eta'] = int(node.attrs['eta'])
            self.operatorRepresentation['act_type'] = int(node.attrs['act_type'])
            self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
            self.operatorRepresentation['dim'] = int(node.attrs['dim'].values)
            self.operatorRepresentation['dim_head'] = int(node.attrs['dim_head'].values)
            self.operatorRepresentation['out_dim'] = int(node.attrs['out_dim'].values)
            self.operatorRepresentation['heads'] = int(node.attrs['heads'].values)

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = [
            'q', 'k', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias', 'wo_weight', 'wo_bias', 'wq_requant_mul',
            'wq_requant_add', 'wq_requant_div', 'wk_requant_mul', 'wk_requant_add', 'wk_requant_div', 'wv_requant_mul',
            'wv_requant_add', 'wv_requant_div', 'kdiv_requant_mul', 'kdiv_requant_add', 'kdiv_requant_div',
            'preattn_requant_mul', 'preattn_requant_add', 'preattn_requant_div', 'postattn_requant_mul',
            'postattn_requant_add', 'postattn_requant_div', 'wo_requant_mul', 'wo_requant_add', 'wo_requant_div'
        ]
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['input_size_Q'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['input_size_KV'] = np.prod(ctxt.lookup(node.inputs[1].name).shape)
        self.operatorRepresentation['q_shape'] = ctxt.lookup(node.inputs[0].name).shape
        self.operatorRepresentation['kv_shape'] = ctxt.lookup(node.inputs[1].name).shape

        return ctxt, True


class iLayerNormParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all(['D' in node.attrs, 'n_levels' in node.attrs, len(node.inputs) == 3, len(node.outputs) == 1])

        if ret:
            self.operatorRepresentation['n_levels'] = int(self._unpack_const(node.attrs['n_levels']))
            self.operatorRepresentation['log2D'] = int(math.log2(self._unpack_const(node.attrs['D'])))

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['data_in', 'weight', 'bias']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['lastDimLength'] = ctxt.lookup(node.inputs[0].name).shape[-1]

        return ctxt, True


class LayerNormParser(iLayerNormParser):

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all(['epsilon' in node.attrs, len(node.inputs) == 3, len(node.outputs) >= 1])

        if ret:
            self.operatorRepresentation['epsilon'] = node.attrs['epsilon']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['data_in', 'weight', 'bias']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['lastDimLength'] = ctxt.lookup(node.inputs[0].name).shape[-1]

        return ctxt, True


class LayerNormGradParser(iLayerNormParser):

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all(['epsilon' in node.attrs, len(node.inputs) == 4, len(node.outputs) == 1])

        if ret:
            self.operatorRepresentation['epsilon'] = node.attrs['epsilon']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['grad_in', 'data_in', 'weight', 'bias']
        outputs = ['grad_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['lastDimLength'] = ctxt.lookup(node.inputs[0].name).shape[-1]

        return ctxt, True


class MatMulParser(NodeParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__()
        self.noBiasHoisting = noBiasHoisting

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([len(node.inputs) >= 2, len(node.outputs) == 1])

        # Assign GEMM-like attributes to be able to reuse same kernel binding
        if ret:
            self.operatorRepresentation['alpha'] = 1
            self.operatorRepresentation['beta'] = 1
            self.operatorRepresentation['transB'] = 0
            self.operatorRepresentation['transA'] = 0

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ret = True

        inputs = ['A', 'B']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            if idx < len(inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        # Store the input and output shapes in the operator representation
        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['A_shape'] = ctxt.lookup(node.inputs[0].name).shape
        self.operatorRepresentation['B_shape'] = ctxt.lookup(node.inputs[1].name).shape
        self.operatorRepresentation['data_out_shape'] = ctxt.lookup(node.outputs[0].name).shape

        # Store the matrix dimensions in the operator representation
        self.operatorRepresentation['M'] = ctxt.lookup(
            node.inputs[0].name).shape[(-2 + self.operatorRepresentation['transA'])]
        self.operatorRepresentation['N'] = ctxt.lookup(
            node.inputs[0].name).shape[(-1 - self.operatorRepresentation['transA'])]
        self.operatorRepresentation['O'] = ctxt.lookup(
            node.inputs[1].name).shape[(-1 - self.operatorRepresentation['transB'])]

        # SCHEREMO: Assert that reduction dimension is the same on both matrices
        ret = ret and (self.operatorRepresentation['N'] == ctxt.lookup(
            node.inputs[1].name).shape[-2 + self.operatorRepresentation['transB']])

        # Check if the batch dimensions are compatible
        self.operatorRepresentation['batch_A'] = np.prod(ctxt.lookup(node.inputs[0].name).shape[:-2])
        self.operatorRepresentation['batch_B'] = np.prod(ctxt.lookup(node.inputs[1].name).shape[:-2])

        self.operatorRepresentation['batch'] = max(self.operatorRepresentation['batch_A'],
                                                   self.operatorRepresentation['batch_B'])

        assert (self.operatorRepresentation["batch_A"] == self.operatorRepresentation["batch_B"]) or (
            self.operatorRepresentation["batch_A"] == 1
        ) or (
            self.operatorRepresentation["batch_B"] == 1
        ), "Incompatible dimensions for input matrices. Broadcasting not yet supported for dimensions larger than 1 on one of the inputs, or equal dimensions between the 2."

        # Create flags for same dimension between each input matrix and the final batch dimension
        self.operatorRepresentation['A_batched'] = (self.operatorRepresentation['batch'] == np.prod(
            ctxt.lookup(node.inputs[0].name).shape[:-2]))
        self.operatorRepresentation['W_batched'] = self.operatorRepresentation['B_batched'] = (
            self.operatorRepresentation['batch'] == np.prod(ctxt.lookup(node.inputs[1].name).shape[:-2]))

        return ctxt, ret


class RQMatMulParser(MatMulParser, RQSParserInterface):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)
        self.noBiasHoisting = noBiasHoisting

    def parseNode(self, node: gs.Node) -> (bool):
        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_matmul = MatMulParser.parseNode(self, node)

        ret = all([
            ret_rqs == True,
            ret_matmul == True,
            len(node.inputs) == 4,
            len(node.outputs) == 1,
        ])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['A', 'B', 'add', 'mul']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

        return newCtxt, ret


# This parser combines Matmul nodes and GEMM nodes to the more general GEMM nodes
class GEMMParser(MatMulParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([
            len(node.inputs) >= 2,
            len(node.outputs) == 1,
        ])

        # This is a GEMM node:
        if ret:

            if 'alpha' in node.attrs:
                self.operatorRepresentation['alpha'] = node.attrs['alpha']
            else:
                self.operatorRepresentation['alpha'] = 1

            if 'beta' in node.attrs:
                self.operatorRepresentation['beta'] = node.attrs['beta']
            else:
                self.operatorRepresentation['beta'] = 1

            if 'transA' in node.attrs:
                self.operatorRepresentation['transA'] = node.attrs['transA']
            else:
                self.operatorRepresentation['transA'] = 0

            if 'transB' in node.attrs:
                self.operatorRepresentation['transB'] = node.attrs['transB']
            else:
                self.operatorRepresentation['transB'] = 0

            if len(node.inputs) == 2 and not self.noBiasHoisting:
                C = gs.Constant(f"{node.name}_C", np.zeros((1,)))
                node.inputs.append(C)

            return True
        # This might be a matmul node -> Cast up
        else:
            return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        # We are a true GEMM
        if ret:
            inputs = ['A', 'B']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                if idx < len(inputs):
                    self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            if len(node.inputs) == 3:
                # Compute bias name and shape if present in the inputs
                self.operatorRepresentation['C'] = newCtxt.lookup(node.inputs[2].name).name
                self.operatorRepresentation['C_shape'] = newCtxt.lookup(node.inputs[2].name).shape

                # Create flag for same dimension between bias matrix and the final batch dimension
                self.operatorRepresentation['C_batched'] = (self.operatorRepresentation['batch'] == np.prod(
                    newCtxt.lookup(node.inputs[2].name).shape[:-2]))

            self.operatorRepresentation['size'] = np.prod(newCtxt.lookup(node.inputs[0].name).shape)

        return newCtxt, ret


class RQGEMMParser(GEMMParser, RQSParserInterface):

    def __init__(self, noBiasHoisting = True):
        self.noBiasHoisting = noBiasHoisting
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_matmul = GEMMParser.parseNode(self, node)

        ret = all([
            ret_rqs == True,
            ret_matmul == True,
            len(node.inputs) == 5,
            len(node.outputs) == 1,
        ])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        # We are a true GEMM
        if ret:
            inputs = ['A', 'B', 'C', 'add', 'mul']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                if idx < len(inputs):
                    self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            if len(node.inputs) == 5:
                self.operatorRepresentation['C'] = newCtxt.lookup(node.inputs[2].name).name
            elif not self.noBiasHoisting:
                values = np.zeros((1))
                zeroTensor = gs.Constant(f'{node.name}_C_Tensor', values = values)
                newCtxt.hoistConstant(zeroTensor)
                self.operatorRepresentation['C'] = f'{node.name}_C_Tensor'

        return newCtxt, ret


class DummyParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = []
        outputs = []
        for i in node.inputs:
            inputs.append(ctxt.lookup(i.name))
        for i in node.outputs:
            outputs.append(ctxt.lookup(i.name))

        self.operatorRepresentation['data_in'] = inputs[0].name
        self.operatorRepresentation['data_out'] = outputs[0].name
        self.operatorRepresentation['size'] = np.prod(inputs[0].shape)

        return ctxt, True


class IntegerDivParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([
            len(node.inputs) >= 2,
            len(node.outputs) == 1,
            'Delta' in node.attrs,
            'eps' in node.attrs,
            'eta' in node.attrs,
        ])

        if ret:
            self.operatorRepresentation['Delta'] = node.attrs['Delta']
            self.operatorRepresentation['eps'] = node.attrs['eps']
            self.operatorRepresentation['eta'] = node.attrs['eta']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ["A", "B"]
        outputs = ["C"]
        for idx, inputNode in enumerate(node.inputs):
            if idx < len(inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['sizeA'] = np.prod(ctxt.lookup(self.operatorRepresentation['A']).shape)
        self.operatorRepresentation['sizeB'] = np.prod(ctxt.lookup(self.operatorRepresentation['B']).shape)

        for idx, (a, b) in enumerate(
                zip(
                    ctxt.lookup(self.operatorRepresentation['A']).shape,
                    ctxt.lookup(self.operatorRepresentation['B']).shape)):
            if a != b:
                self.operatorRepresentation['nomStep'] = np.prod(
                    ctxt.lookup(self.operatorRepresentation['A']).shape[idx:])
                self.operatorRepresentation['denomStep'] = np.prod(
                    ctxt.lookup(self.operatorRepresentation['B']).shape[idx:])
                break

        return ctxt, True


class PowParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        return node.op == 'Pow' and len(node.inputs) == 2 and len(node.outputs) == 1

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        # Lookup both inputs (data and exponent)
        data_in = ctxt.lookup(node.inputs[0].name)
        exponent_tensor = ctxt.lookup(node.inputs[1].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['exponent'] = exponent_tensor.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))

        return ctxt, True


class DivParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) == 2, len(node.outputs) == 1])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ["input1", "input2"]
        outputs = ["output"]
        for idx, inputNode in enumerate(node.inputs):
            if idx < len(inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(self.operatorRepresentation['input1']).shape)

        return ctxt, True


class RQIntegerDivParser(IntegerDivParser, RQSParserInterface):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = RQSParserInterface.parseNode(self, node)

        if ret:
            ret = IntegerDivParser.parseNode(self, node)

        wellFormed = all([
            len(node.inputs) == 5,
        ])

        if ret:
            return wellFormed

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        inputs = ["A", "B", "requant_mul", "requant_add", "requant_div"]
        outputs = ["C"]
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

        return newCtxt, ret


class DebugParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) == 1, len(node.outputs) == 1],)

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)

        wellFormed = False
        if len(data_in.shape) == 4:
            wellFormed = True
            self.operatorRepresentation['batch'] = data_in.shape[0]
            if channels_first:
                self.operatorRepresentation['dim_im_in_ch'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
                self.operatorRepresentation['dim_im_out_ch'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_x'] = data_out.shape[2]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[3]
            else:
                self.operatorRepresentation['dim_im_in_x'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_ch'] = data_in.shape[3]
                self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]
                self.operatorRepresentation['dim_im_out_ch'] = data_out.shape[3]

        if len(data_in.shape) == 3:
            wellFormed = True
            self.operatorRepresentation['batch'] = data_in.shape[0]
            self.operatorRepresentation['dim_im_in_ch'] = 1
            self.operatorRepresentation['dim_im_in_x'] = data_in.shape[1]
            self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
            self.operatorRepresentation['dim_im_out_ch'] = 1
            self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
            self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]

        if len(data_in.shape) == 2:
            wellFormed = True
            self.operatorRepresentation['batch'] = data_in.shape[0]
            self.operatorRepresentation['dim_im_in_x'] = 1
            self.operatorRepresentation['dim_im_out_x'] = 1
            self.operatorRepresentation['dim_im_in_ch'] = 1
            self.operatorRepresentation['dim_im_out_ch'] = 1
            self.operatorRepresentation['dim_im_in_y'] = data_in.shape[1]
            self.operatorRepresentation['dim_im_out_y'] = data_out.shape[1]

        return ctxt, wellFormed


class GenericMaxPool2DParser(MaxPool2DParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                all([pad == 0 for pad in self.operatorRepresentation['pads']]), self.operatorRepresentation['ceil_mode']
                == 0
            ],)

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret


class GenericConv1DParser(Conv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['group'] == 1,
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                self.operatorRepresentation['pads'][0] == 0,
                all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) > 2:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = 1
            else:
                self.operatorRepresentation["has_bias"] = 0
                self.operatorRepresentation["bias"] = "NULL"

            for idx, inputNode in enumerate(node.inputs):
                if idx >= len(inputs):
                    raise IndexError(
                        f"Index {idx} out of range for inputs of length {len(inputs)} in node {inputNode.name}")

                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class GenericDWConv1DParser(Conv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                self.operatorRepresentation['pads'][0] == 0,
                all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            if self.operatorRepresentation['group'] == self.operatorRepresentation['ch_im_in']:
                return newCtxt, True

        return ctxt, False


class GenericConv2DParser(Conv2DParser):

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
                self.operatorRepresentation['pads'][0] == 0,
                all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) > 2:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = "true"
            else:
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True
        else:
            return ctxt, False

        assert len(node.inputs
                  ) == 2, f'Supports only parsing 2 input tensors, data_in and weight. Received: {len(node.inputs)}'
        for node, sym_name in zip(node.inputs, ['data_in', 'weight']):
            self.operatorRepresentation[sym_name] = ctxt.lookup(node.name).name

        return newCtxt, True


class GenericDWConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                self.operatorRepresentation['pads'][0] == 0,
                all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) > 2:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = "true"
            else:
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            if self.operatorRepresentation['group'] == self.operatorRepresentation['ch_im_in']:
                return newCtxt, True

        return ctxt, False


class GenericGEMMParser(GEMMParser):

    def __init__(self, noBiasHoisting = False):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            # Try to scale A offline if possible, else fail
            if not self.operatorRepresentation['alpha'].is_integer():
                nameA = self.operatorRepresentation['A']
                if newCtxt.is_global(nameA) and isinstance(newCtxt.lookup(nameA), ConstantBuffer):
                    A = newCtxt.lookup(nameA)
                    npA = np.asarray(A.values).reshape(A.shape)
                    newA = npA * self.operatorRepresentation['beta']
                    newCtxt.globalObjects[nameA].values = newA
                    self.operatorRepresentation['alpha'] = 1.0
                else:
                    return newCtxt, False
            # Try to scale B offline if possible, else fail
            if not self.operatorRepresentation['beta'].is_integer():
                nameB = self.operatorRepresentation['B']
                if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                    B = newCtxt.lookup(nameB)
                    npB = np.asarray(B.values).reshape(B.shape)
                    newB = npB * self.operatorRepresentation['beta']
                    newCtxt.globalObjects[nameB].values = newB
                    self.operatorRepresentation['beta'] = 1.0
                else:
                    return newCtxt, False

            self.operatorRepresentation['alpha'] = int(self.operatorRepresentation['alpha'])
            self.operatorRepresentation['beta'] = int(self.operatorRepresentation['beta'])
            return newCtxt, True

        return ctxt, False


class RQAddParser(AddParser):

    def parseNode(self, node: gs.Node) -> bool:

        if not super().parseNode(node):
            return False

        ret = all([
            'rqs1_mul' in node.attrs,
            'rqs1_add' in node.attrs,
            'rqs1_div' in node.attrs,
            'rqs1_signed' in node.attrs,
            any(['rqs1_n_levels' in node.attrs, 'rqs1_n_levels_out' in node.attrs]),
            'rqs2_mul' in node.attrs,
            'rqs2_add' in node.attrs,
            'rqs2_div' in node.attrs,
            'rqs2_signed' in node.attrs,
            any(['rqs2_n_levels' in node.attrs, 'rqs2_n_levels_out' in node.attrs]),
            'rqsOut_mul' in node.attrs,
            'rqsOut_add' in node.attrs,
            'rqsOut_div' in node.attrs,
            'rqsOut_signed' in node.attrs,
            any(['rqsOut_n_levels' in node.attrs, 'rqsOut_n_levels_out' in node.attrs]),
        ])

        if ret:
            if 'rqs1_n_levels' in node.attrs:
                self.operatorRepresentation['rqs1_n_levels'] = int(node.attrs['rqs1_n_levels'].values)
            else:
                self.operatorRepresentation['rqs1_n_levels'] = int(node.attrs['rqs1_n_levels_out'].values)
            self.operatorRepresentation['rqs1_mul'] = int(node.attrs['rqs1_mul'])
            self.operatorRepresentation['rqs1_add'] = int(node.attrs['rqs1_add'])
            self.operatorRepresentation['rqs1_signed'] = int(node.attrs['rqs1_signed'].values)
            self.operatorRepresentation['rqs1_log2D'] = int(math.log2(node.attrs['rqs1_div'].values))

            if 'rqs2_n_levels' in node.attrs:
                self.operatorRepresentation['rqs2_n_levels'] = int(node.attrs['rqs2_n_levels'].values)
            else:
                self.operatorRepresentation['rqs2_n_levels'] = int(node.attrs['rqs2_n_levels_out'].values)
            self.operatorRepresentation['rqs2_mul'] = int(node.attrs['rqs2_mul'])
            self.operatorRepresentation['rqs2_add'] = int(node.attrs['rqs2_add'])
            self.operatorRepresentation['rqs2_signed'] = int(node.attrs['rqs2_signed'].values)
            self.operatorRepresentation['rqs2_log2D'] = int(math.log2(node.attrs['rqs2_div'].values))

            if 'rqsOut_n_levels' in node.attrs:
                self.operatorRepresentation['rqsOut_n_levels'] = int(node.attrs['rqsOut_n_levels'].values)
            else:
                self.operatorRepresentation['rqsOut_n_levels'] = int(node.attrs['rqsOut_n_levels_out'].values)
            self.operatorRepresentation['rqsOut_mul'] = int(node.attrs['rqsOut_mul'])
            self.operatorRepresentation['rqsOut_add'] = int(node.attrs['rqsOut_add'])
            self.operatorRepresentation['rqsOut_signed'] = int(node.attrs['rqsOut_signed'].values)
            self.operatorRepresentation['rqsOut_log2D'] = int(math.log2(node.attrs['rqsOut_div'].values))

        return ret


class QuantParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([
            'scale' in node.attrs, 'zero_point' in node.attrs, 'bit_width' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.operatorRepresentation['scale'] = float(node.attrs['scale'])
            self.operatorRepresentation['zero_point'] = float(node.attrs['zero_point'])
            self.operatorRepresentation['bit_width'] = int(node.attrs['bit_width'])

            # Handle optional signed attribute
            if 'signed' in node.attrs:
                self.operatorRepresentation['signed'] = bool(node.attrs['signed'])
            else:
                self.operatorRepresentation['signed'] = True  # Default to signed

            # Calculate min and max values based on bit_width and signed
            bit_width_int = self.operatorRepresentation['bit_width']
            if self.operatorRepresentation['signed']:
                self.operatorRepresentation['min_val'] = -(2**(bit_width_int - 1))
                self.operatorRepresentation['max_val'] = (2**(bit_width_int - 1)) - 1
            else:
                self.operatorRepresentation['min_val'] = 0
                self.operatorRepresentation['max_val'] = (2**bit_width_int) - 1

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)

        return ctxt, True


class DequantParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            'scale' in node.attrs, 'zero_point' in node.attrs, 'bit_width' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.operatorRepresentation['scale'] = float(node.attrs['scale'])
            self.operatorRepresentation['zero_point'] = float(node.attrs['zero_point'])
            self.operatorRepresentation['bit_width'] = int(node.attrs['bit_width'])

            self.operatorRepresentation['signed'] = bool(node.attrs['signed'])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)

        return ctxt, True


class SoftmaxCrossEntropyLossParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) == 2, len(node.outputs) == 1])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        logits = ctxt.lookup(node.inputs[0].name)
        labels = ctxt.lookup(node.inputs[1].name)
        log_prob = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['logits'] = logits.name
        self.operatorRepresentation['labels'] = labels.name
        self.operatorRepresentation['log_prob'] = log_prob.name
        self.operatorRepresentation['batch'] = logits.shape[0]
        self.operatorRepresentation['num_classes'] = logits.shape[1]

        return ctxt, True


class SoftmaxCrossEntropyLossGradParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) == 2, len(node.outputs) == 1])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        log_prob = ctxt.lookup(node.inputs[0].name)
        labels = ctxt.lookup(node.inputs[1].name)
        grad = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['log_prob'] = log_prob.name
        self.operatorRepresentation['labels'] = labels.name
        self.operatorRepresentation['grad'] = grad.name
        self.operatorRepresentation['batch'] = log_prob.shape[0]  # RW: used for tiling
        self.operatorRepresentation['total_batch'] = log_prob.shape[0]  # RW: total batch num for normalization
        self.operatorRepresentation['num_classes'] = log_prob.shape[1]

        return ctxt, True


class SGDParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        ret = all([len(node.inputs) == 2, len(node.outputs) == 1, 'lr' in node.attrs])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        weight = ctxt.lookup(node.inputs[0].name)
        grad = ctxt.lookup(node.inputs[1].name)
        weight_updated = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['weight'] = weight.name
        self.operatorRepresentation['grad'] = grad.name
        self.operatorRepresentation['weight_updated'] = weight_updated.name
        self.operatorRepresentation['size'] = np.prod(weight.shape)
        self.operatorRepresentation['lr'] = node.attrs['lr']

        return ctxt, True


class BatchNormParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        # Verify the attributes (epsilon is mandatory, momentum and training_mode  are optional)
        if 'epsilon' not in node.attrs:
            return False
        # Common Inputs: 5 (X, scale, B, mean, var)
        if len(node.inputs) < 5:
            return False

        # Save the attributes, default values are provided if not present
        self.operatorRepresentation['epsilon'] = node.attrs.get('epsilon', 1e-5)
        self.operatorRepresentation['momentum'] = node.attrs.get('momentum', 0.9)
        self.operatorRepresentation['training_mode'] = node.attrs.get('training_mode', 0)

        return True

    def parseNodeCtxt(self, ctxt, node: gs.Node, channels_first: bool = True):
        inputs = ['data_in', 'scale', 'bias', 'mean', 'variance']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs[:5]):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

        # Output (Y)
        self.operatorRepresentation[outputs[0]] = ctxt.lookup(node.outputs[0].name).name

        input_shape = ctxt.lookup(node.inputs[0].name).shape
        # Save input shape information
        self.operatorRepresentation['batch_size'] = input_shape[0]
        self.operatorRepresentation['channel_size'] = input_shape[1]
        self.operatorRepresentation['window_size'] = input_shape[2]

        return ctxt, True


class ConvTransposeParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        # Extract ONNX attributes with defaults
        strides = node.attrs.get('strides', [1])

        pads = node.attrs.get('pads', [0, 0])
        kernel_shape = node.attrs.get('kernel_shape', None)
        dilations = node.attrs.get('dilations', [1])
        group = node.attrs.get('group', 1)

        # Check for required attributes
        wellFormed = (kernel_shape is not None and len(node.outputs) == 1)
        if wellFormed:
            self.operatorRepresentation['strides'] = strides
            self.operatorRepresentation['pads'] = pads
            self.operatorRepresentation['kernel_shape'] = kernel_shape
            self.operatorRepresentation['dilations'] = dilations
            self.operatorRepresentation['group'] = group
            self.operatorRepresentation['nodeName'] = node.name
            self.operatorRepresentation['nodeOp'] = node.op
        return wellFormed

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node, channels_first: bool = True):
        # Register buffer names for codegen
        self.operatorRepresentation['data_in'] = node.inputs[0].name
        self.operatorRepresentation['weight'] = node.inputs[1].name
        self.operatorRepresentation['data_out'] = node.outputs[0].name
        if len(node.inputs) == 3:
            self.operatorRepresentation['bias'] = node.inputs[2].name
            self.operatorRepresentation['has_bias'] = "true"
        else:
            self.operatorRepresentation['has_bias'] = "false"
        # Get output shape from context
        data_out = ctxt.lookup(node.outputs[0].name)
        out_shape = data_out.shape
        if len(out_shape) == 3:
            self.operatorRepresentation['dim_im_out_x'] = out_shape[2]
        elif len(out_shape) == 4:
            self.operatorRepresentation['dim_im_out_x'] = out_shape[2]
            self.operatorRepresentation['dim_im_out_y'] = out_shape[3]

        stride_x, stride_y = 1, 1
        if "strides" in node.attrs:
            stride_y = node.attrs["strides"][0]
            stride_x = node.attrs["strides"][1] if len(node.attrs["strides"]) > 1 else stride_y
        self.operatorRepresentation["stride_y"] = stride_y
        self.operatorRepresentation["stride_x"] = stride_x

        if "kernel_shape" in node.attrs:
            kernel_shape = node.attrs["kernel_shape"]
            kernel_shape_x = kernel_shape[0]
            # For 2D, kernel_shape may have two elements
            kernel_shape_y = kernel_shape[1] if len(kernel_shape) > 1 else kernel_shape_x
        else:
            kernel_shape_x = 1
            kernel_shape_y = 1

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        in_shape = data_in.shape
        out_shape = data_out.shape

        self.operatorRepresentation['ch_im_in'] = in_shape[1]
        self.operatorRepresentation['dim_im_in_y'] = in_shape[2]
        self.operatorRepresentation['ch_im_out'] = out_shape[1]
        self.operatorRepresentation['dim_im_out_y'] = out_shape[2]

        self.operatorRepresentation[
            'batchOffsetIn'] = self.operatorRepresentation['ch_im_in'] * self.operatorRepresentation['dim_im_in_y']
        self.operatorRepresentation[
            'batchOffsetOut'] = self.operatorRepresentation['ch_im_out'] * self.operatorRepresentation['dim_im_out_y']
        return ctxt, True


class ConvTranspose1DParser(ConvTransposeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        # 1D ConvTranspose expects 3D input/output and 3D weight
        wellFormed = super().parseNode(node)
        ret = False
        if wellFormed:
            ret = all([
                # Make sure strides are 2D
                len(node.attrs['strides']) == 1,
                len(node.attrs['pads']) == 2,
                len(node.attrs['dilations']) == 1,
            ])
        if ret:

            self.operatorRepresentation['kernel_shape'] = node.attrs['kernel_shape']
            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['padding_y'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            data_in = newCtxt.lookup(node.inputs[0].name)
            data_out = newCtxt.lookup(node.outputs[0].name)
            in_shape = data_in.shape
            out_shape = data_out.shape
            self.operatorRepresentation['batch'] = in_shape[0]
            self.operatorRepresentation['ch_im_in'] = in_shape[1]
            self.operatorRepresentation['dim_im_in_y'] = in_shape[2]
            self.operatorRepresentation['ch_im_out'] = out_shape[1]
            self.operatorRepresentation['dim_im_out_y'] = out_shape[2]
            self.operatorRepresentation[
                "batchOffsetIn"] = self.operatorRepresentation["ch_im_in"] * self.operatorRepresentation["dim_im_in_y"]
            self.operatorRepresentation["batchOffsetOut"] = self.operatorRepresentation[
                "ch_im_out"] * self.operatorRepresentation["dim_im_out_y"]
            return newCtxt, True
        return ctxt, False


class SqrtParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        return node.op == 'Sqrt' and len(node.inputs) == 1 and len(node.outputs) == 1

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))

        return ctxt, True
