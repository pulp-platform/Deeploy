# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class PULP2DConvTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(operatorRepresentation['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = 2 * 8 * (operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_x'] *
                              operatorRepresentation['dim_kernel_y'])
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DConvTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULP2DDWConvTemplate(PULP2DConvTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(operatorRepresentation['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []


class PULP1DConvTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(operatorRepresentation['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        operatorRepresentation['pad_x_left'] = operatorRepresentation['pads'][0]
        operatorRepresentation['pad_x_right'] = operatorRepresentation['pads'][1]
        operatorRepresentation['stride_x'] = operatorRepresentation['strides'][0]

        return ctxt, operatorRepresentation, []

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = 8 * 2 * operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_y']
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP1DConvTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)
        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULP1DDWConvTemplate(PULP1DConvTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


PULPConv2D_8_Template = PULP2DConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>

<%
operatorString = ''
if dim_kernel_x == 1 and dim_kernel_y == 1:
    operatorString = 'pointwise'
else:
    operatorString = 'conv'
operatorString = 'conv'
%>

pulp_nn_${operatorString}${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, ${mul}, ${add}, 1, ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out}, ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_y}, ${stride_x}, 1, 1);
""")

PULPDWConv2D_8_Template = PULP2DDWConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
pulp_nn_depthwise${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, NULL, ${mul}, ${add}, 1, ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out}, ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_y}, ${stride_x}, 1, 1);
""")

PULPConv1D_8_Template = PULP1DConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>

pulp_nn_conv${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, ${mul}, ${add}, 1, ${log2D}, 1, ${dim_im_in_y}, ${ch_im_in}, 1, ${dim_im_out_y}, ${ch_im_out}, 1, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, 0, 0, 1, ${stride_y}, 1, 1);
""")

PULPDWConv1D_8_Template = PULP1DDWConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
pulp_nn_depthwise${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, NULL, ${mul}, ${add}, 1, ${log2D}, 1, ${dim_im_in_y}, ${ch_im_in}, 1, ${dim_im_out_y}, ${ch_im_out}, 1, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, 0, 0, 1, ${stride_y}, 1, 1);
""")
