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
                # Current PULP kernel only supports grouping of 1
                self.operatorRepresentation['group'] == 1,

                # Make sure padding is square
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],

                # Check number of inputs
                # 2 inputs if no bias, 3 if layer has bias
                len(node.inputs) in [2, 3],
            ])

            # Extract additional attributes
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # Set inputs names
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) == 2:
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"
            else:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = "true"

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPFPDWConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        # Parse root conv 2D information
        wellFormed = super().parseNode(node)

        if wellFormed:
            # Check if the node is a depthwise convolution
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],

                # Check number of inputs
                # 2 inputs if no bias, 3 if layer has bias
                len(node.inputs) in [2, 3],
            ])

            # Extract additional attributes
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        # Parse node context for 2D conv
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # Define input names
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) == 2:
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"
            else:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = "true"

            # Map input nodes to operator representation
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            # Check if DW
            if self.operatorRepresentation['group'] == self.operatorRepresentation['ch_im_in']:
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

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        if not (self.operatorRepresentation['M'] == 1 and self.operatorRepresentation['batch'] >= 8):
            return ctxt, False

        return newCtxt, True


class PULPTallGEMMParser(PULPGEMMParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        ret = all([
            self.operatorRepresentation['batch'] < 8,
            self.operatorRepresentation['M'] >= 8,
            self.operatorRepresentation['M'] % 8 < self.operatorRepresentation['O'] % 8,
        ])

        if not ret:
            return ctxt, False

        return newCtxt, True

class PULPConvTrans2DParser(PULPFPConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)
    
    def parseNode(self, node: gs.Node) -> bool:
        """Override to recognize ConvGradX instead of Conv"""
        # Temporarily change op to Conv for parent parsing
        original_op = node.op
        if node.op == 'ConvGradX':
            node.op = 'Conv'
        
        # Call parent parseNode
        wellFormed = super().parseNode(node)
        
        # Restore original op
        node.op = original_op
        
        # Additional validation for ConvGradX
        if wellFormed and original_op == 'ConvGradX':
            # ConvGradX should have 2 inputs: output_grad and weight
            return len(node.inputs) == 2
        
        return wellFormed
    
    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        """Override for ConvGradX - swap input/output semantics"""
        
        if node.op == 'ConvGradX':
            # For ConvGradX: inputs are [output_grad, weight], output is input_grad
            # But parent expects: inputs are [input, weight], output is output
            # So we need to swap the semantics
            
            # Temporarily swap input/output for parent parsing
            output_grad_name = node.inputs[0].name
            input_grad_name = node.outputs[0].name
            
            # Get tensors
            output_grad = ctxt.lookup(output_grad_name)
            weight = ctxt.lookup(node.inputs[1].name)
            
            # Create a temporary input tensor with output_grad's info as if it's the output
            # and output tensor with input_grad's info as if it's the input
            temp_input = node.inputs[0]
            temp_output = node.outputs[0]
            
            # Swap
            node.inputs[0] = temp_output
            node.outputs[0] = temp_input
            
            # Call parent
            newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
            
            # Restore
            node.inputs[0] = temp_input
            node.outputs[0] = temp_output
            
            if ret:
                # Fix the tensor names for ConvGradX
                self.operatorRepresentation['data_in'] = output_grad_name
                self.operatorRepresentation['data_out'] = input_grad_name
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"
                
            return newCtxt, ret
        else:
            return super().parseNodeCtxt(ctxt, node, channels_first)

class PULPDWConvTrans2DParser(PULPFPDWConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)
    
    def parseNode(self, node: gs.Node) -> bool:
        """Override to recognize ConvGradX instead of Conv"""
        # Temporarily change op to Conv for parent parsing
        original_op = node.op
        if node.op == 'ConvGradX':
            node.op = 'Conv'
        
        # Call parent parseNode
        wellFormed = super().parseNode(node)
        
        # Restore original op
        node.op = original_op
        
        # Additional validation for ConvGradX
        if wellFormed and original_op == 'ConvGradX':
            # ConvGradX should have 2 inputs: output_grad and weight
            return len(node.inputs) == 2
        
        return wellFormed
    
    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        """Override for ConvGradX - swap input/output semantics"""
        
        if node.op == 'ConvGradX':
            # For ConvGradX: inputs are [output_grad, weight], output is input_grad
            # Temporarily swap input/output for parent parsing
            output_grad_name = node.inputs[0].name
            input_grad_name = node.outputs[0].name
            
            # Swap
            temp_input = node.inputs[0]
            temp_output = node.outputs[0]
            node.inputs[0] = temp_output
            node.outputs[0] = temp_input
            
            # Call parent
            newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
            
            # Restore
            node.inputs[0] = temp_input
            node.outputs[0] = temp_output
            
            if ret:
                # Fix the tensor names for ConvGradX
                self.operatorRepresentation['data_in'] = output_grad_name
                self.operatorRepresentation['data_out'] = input_grad_name
                self.operatorRepresentation["weight"] = ctxt.lookup(node.inputs[1].name).name
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"
                
            return newCtxt, ret
        else:
            return super().parseNodeCtxt(ctxt, node, channels_first)


class PULPConvGradW2DParser(PULPFPConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        """Parse ConvGradW - need custom logic for input dimensions"""
        
        if not self.parseNode(node):
            return ctxt, False
        
        # Get input tensors
        grad_out_tensor = ctxt.lookup(node.inputs[0].name)
        data_in_tensor = ctxt.lookup(node.inputs[1].name)
        
        # Extract batch size
        batch = grad_out_tensor.shape[0]
        
        # Extract dimensions
        C_out, H_out, W_out = grad_out_tensor.shape[1], grad_out_tensor.shape[2], grad_out_tensor.shape[3]
        C_in, H_in, W_in = data_in_tensor.shape[1], data_in_tensor.shape[2], data_in_tensor.shape[3]
    
        # Store batch size
        self.operatorRepresentation['batch'] = batch
        
        # Store dimensions
        self.operatorRepresentation['ch_im_out'] = C_out
        self.operatorRepresentation['dim_im_out_x'] = W_out
        self.operatorRepresentation['dim_im_out_y'] = H_out
        self.operatorRepresentation['ch_im_in'] = C_in
        self.operatorRepresentation['dim_im_in_x'] = W_in
        self.operatorRepresentation['dim_im_in_y'] = H_in
        
        # Store kernel dimensions
        self.operatorRepresentation['dim_kernel_y'] = self.operatorRepresentation['kernel_shape'][0]
        self.operatorRepresentation['dim_kernel_x'] = self.operatorRepresentation['kernel_shape'][1]
        
        # Store strides
        self.operatorRepresentation['stride_y'] = self.operatorRepresentation['strides'][0]
        self.operatorRepresentation['stride_x'] = self.operatorRepresentation['strides'][1]
        
        # Set tensor names and types
        self.operatorRepresentation['grad_out'] = node.inputs[0].name
        self.operatorRepresentation['grad_out_type'] = grad_out_tensor._type
        self.operatorRepresentation['data_in'] = node.inputs[1].name
        self.operatorRepresentation['data_in_type'] = data_in_tensor._type
        self.operatorRepresentation['weight'] = node.outputs[0].name
        self.operatorRepresentation['weight_type'] = grad_out_tensor._type  # Same as grad_out
        
        # No bias for ConvGradW
        self.operatorRepresentation['has_bias'] = 'false'
        self.operatorRepresentation['bias'] = 'NULL'
        
        return ctxt, True

class PULPConvGradB2DParser(PULPFPConv2DParser):

    def __init__(self):
        self.operatorRepresentation = {}

    def parseNode(self, node: gs.Node) -> bool:
        """Parse ConvGradB node attributes"""
        
        # Check basic structure
        if node.op != 'ConvGradB':
            return False
        
        if len(node.inputs) != 1:  # only output_grad
            return False
        
        if len(node.outputs) != 1:  # bias_grad
            return False
        
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        """Parse ConvGradB node context"""

        # For ConvGradB, the inputs are:
        # inputs[0]: output_grad [N, C_out, H_out, W_out] (NCHW)
        # output:    bias_grad [C_out]

        # Get tensors from context
        output_grad_tensor = ctxt.lookup(node.inputs[0].name)
        
        # Extract batch size and dimensions (NCHW)
        batch = output_grad_tensor.shape[0]
        C_out = output_grad_tensor.shape[1]
        H_out = output_grad_tensor.shape[2]
        W_out = output_grad_tensor.shape[3]
        
        # Store batch size
        self.operatorRepresentation['batch'] = batch
        
        # Store dimensions
        self.operatorRepresentation['ch_im_out'] = C_out
        self.operatorRepresentation['dim_im_out_x'] = W_out
        self.operatorRepresentation['dim_im_out_y'] = H_out
        
        # Dummy kernel_shape for computeOps (ConvGradB doesn't use kernels)
        self.operatorRepresentation['kernel_shape'] = [1, 1]
        self.operatorRepresentation['ch_im_in'] = 1  # Dummy value
        
        # Set tensor names and types
        self.operatorRepresentation['grad_out'] = node.inputs[0].name
        self.operatorRepresentation['grad_out_type'] = output_grad_tensor._type
        self.operatorRepresentation['bias'] = node.outputs[0].name
        self.operatorRepresentation['bias_type'] = output_grad_tensor._type  # Same type as grad_out
        
        return ctxt, True