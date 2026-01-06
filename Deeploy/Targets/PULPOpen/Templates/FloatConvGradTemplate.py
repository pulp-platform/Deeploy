# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class PULP2DFloatConvGradWIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # For ConvGradW, im2col buffer stores im2row transformed input
        # Size: H_out * W_out * kernel_h * kernel_w * C_in * sizeof(float)
        im2col_dim = (operatorRepresentation["data_in_type"].typeWidth // 8) * \
                     operatorRepresentation['dim_im_out_x'] * operatorRepresentation['dim_im_out_y'] * \
                     operatorRepresentation['ch_im_in'] * \
                     operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        im2col_name = operatorRepresentation['nodeName'] + "_buffer"

        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DFloatConvGradWIm2ColTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULP2DFloatConvGradXIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = (operatorRepresentation["data_in_type"].typeWidth // 8) * \
                     operatorRepresentation['dim_im_in_x'] * operatorRepresentation['dim_im_in_y'] * \
                     operatorRepresentation['ch_im_out'] * \
                     operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        im2col_name = operatorRepresentation['nodeName'] + "_im2col_buffer"

        # Block transpose buffer for weight transformation
        # Size: C_in * (P * Q * C_out) for transposed weight matrix
        # This transforms W[C_out, C_in, P, Q] into W_T[C_in, C_out*P*Q] format
        bt_dim = (operatorRepresentation["weight_type"].typeWidth // 8) * \
                 operatorRepresentation['ch_im_in'] * operatorRepresentation['ch_im_out'] * \
                 operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        bt_name = operatorRepresentation['nodeName'] + "_bt_buffer"

        return [(im2col_name, im2col_dim), (bt_name, bt_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        buffers = PULP2DFloatConvGradXIm2ColTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)

        im2col_name, im2col_dim = buffers[0]
        bt_name, bt_dim = buffers[1]

        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)
        ctxt.hoistTransientBuffer(bt_name, bt_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        operatorRepresentation['btBuffer'] = bt_name
        operatorRepresentation['btBufferSize'] = bt_dim

        return ctxt, operatorRepresentation, [im2col_name, bt_name]



referenceConvGradX2DTemplate = NodeTemplate("""
// 2D FP ConvGradX (dX) NCHW/CHW trainlib naive (Name: ${nodeName}, Op: ${nodeOp})
${data_in_type.typeName}  ref_${data_out}_${data_in} = ${data_in};   // dY
${weight_type.typeName}   ref_${data_out}_${weight}  = ${weight};    // W
${data_out_type.typeName} ref_${data_out}_out        = ${data_out};  // dX

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradX2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_CHW_tiled(
        ref_${data_out}_${data_in},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${data_out}_${weight},
        ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${data_out}_out,
        ${dim_im_in_x}, ${dim_im_in_y},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
        ${offset_grad_in_h}, ${offset_grad_in_w},
        ${offset_grad_out_h}, ${offset_grad_out_w}

    );

    ref_${data_out}_${data_in} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${data_out}_out        += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
}
""")

# referenceConvGradX2DIm2ColTemplate = PULP2DFloatConvGradXIm2ColTemplate("""
# // 2D FP ConvGradX (dX) NCHW/CHW using pulp-trainlib im2col (Name: ${nodeName}, Op: ${nodeOp})
# ${data_in_type.typeName}  ref_${data_out}_${data_in} = ${data_in};   // dY
# ${weight_type.typeName}   ref_${data_out}_${weight}  = ${weight};    // W
# ${data_out_type.typeName} ref_${data_out}_out        = ${data_out};  // dX

# for (uint32_t n=0; n<${batch}; ++n) {
#     PULP_ConvGradX2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_CHW_Im2Col(
#         ref_${data_out}_${data_in},
#         ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
#         ref_${data_out}_${weight},
#         ${ch_im_in},
#         ${dim_kernel_x}, ${dim_kernel_y},
#         ${stride_x}, ${stride_y},
#         ref_${data_out}_out,
#         ${dim_im_in_x}, ${dim_im_in_y},
#         ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
#         ${ctxtBuffer}, ${ctxtBufferSize},
#         ${btBuffer}, ${btBufferSize}
#     );

#     ref_${data_out}_${data_in} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
#     ref_${data_out}_out        += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
# }
# """)

referenceConvGradX2DIm2ColTiledTemplate = PULP2DFloatConvGradXIm2ColTemplate("""
// 2D FP ConvGradX (dX) NCHW/CHW using tile-aware Im2Col (Name: ${nodeName}, Op: ${nodeOp})
PULP_ConvGradX2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_CHW_Im2Col_tiled(
    ${data_in},                                   // dY tile pointer (L1)
    ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},  // dY tile dims
    ${weight},                                    // W
    ${ch_im_in},
    ${dim_kernel_x}, ${dim_kernel_y},
    ${stride_x}, ${stride_y},
    ${data_out},                                  // dX tile pointer (L1)
    ${dim_im_in_x}, ${dim_im_in_y},              // dX tile dims
    ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${offset_grad_in_h}, ${offset_grad_in_w},
    ${offset_grad_out_h}, ${offset_grad_out_w},
    ${ctxtBuffer}, ${ctxtBufferSize},
    ${btBuffer}, ${btBufferSize}
);
""")


referenceConvGradW2DTemplate = NodeTemplate("""
// 2D FP ConvGradW NCHW using pulp-trainlib naive (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${weight}_${data_in} = ${data_in};
${weight_type.typeName} ref_${weight}_out = ${weight};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_CHW(
        ref_${weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${weight}_out,
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${weight}_${data_in} += ${ch_im_in} * ${dim_im_in_y} * ${dim_im_in_x};
}
""")

referenceConvGradW2DIm2ColTemplate = PULP2DFloatConvGradWIm2ColTemplate("""
// 2D FP ConvGradW NCHW using pulp-trainlib naive (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${weight}_${data_in} = ${data_in};
${weight_type.typeName} ref_${weight}_out = ${weight};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_CHW_Im2Col(
        ref_${weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${weight}_out,
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
        ${ctxtBuffer}, ${ctxtBufferSize}
    );

    ref_${weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${weight}_${data_in} += ${ch_im_in} * ${dim_im_in_y} * ${dim_im_in_x};
}
""")

referenceDWConvGradW2DTemplate = NodeTemplate("""
// 2D FP DW ConvGradW NCHW (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${weight}_${data_in} = ${data_in};
${weight_type.typeName} ref_${weight}_out = ${weight};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_DWConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_CHW(
        ref_${weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${weight}_out,
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
    ref_${weight}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
}
""")

referenceDWConvGradX2DTiledTemplate = NodeTemplate("""
// 2D FP DW ConvGradX (dX) CHW tiled (Name: ${nodeName}, Op: ${nodeOp})
${data_in_type.typeName}  ref_${data_out}_${data_in} = ${data_in};   // dY
${weight_type.typeName}   ref_${data_out}_${weight}  = ${weight};    // W 
${data_out_type.typeName} ref_${data_out}_out        = ${data_out};  // dX

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_DWConvGradX2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_CHW_tiled(
        ref_${data_out}_${data_in},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${data_out}_${weight},
        ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${data_out}_out,
        ${dim_im_in_x}, ${dim_im_in_y},
        ${padding_x_left}, ${padding_x_right}, ${padding_y_top}, ${padding_y_bottom},
        ${offset_grad_in_h}, ${offset_grad_in_w},
        ${offset_grad_out_h}, ${offset_grad_out_w}
    );

    ref_${data_out}_${data_in} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${data_out}_out        += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
}
""")


referencePWConvGradW2DTemplate = NodeTemplate("""
// 2D FP Pointwise ConvGradW (1x1) NCHW using pulp-trainlib pw interface (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${weight}_${data_in} = ${data_in};
${weight_type.typeName} ref_${weight}_out = ${weight};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_PWConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_CHW(
        ref_${weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ref_${weight}_out
    );

    ref_${weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${weight}_${data_in} += ${ch_im_in} * ${dim_im_in_y} * ${dim_im_in_x};
}
""")