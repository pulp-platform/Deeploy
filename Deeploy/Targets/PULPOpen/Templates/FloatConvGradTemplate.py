# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation




referenceConvGradX2DTemplate = NodeTemplate("""
// 2D FP ConvGradX (dX) NCHW/CHW trainlib naive (Name: ${nodeName}, Op: ${nodeOp})
${data_in_type.typeName}  ref_${data_out}_${data_in} = ${data_in};   // dY
${weight_type.typeName}   ref_${data_out}_${weight}  = ${weight};    // W
${data_out_type.typeName} ref_${data_out}_out        = ${data_out};  // dX

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradX2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_CHW(
        ref_${data_out}_${data_in},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${data_out}_${weight},
        ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${data_out}_out,
        ${dim_im_in_x}, ${dim_im_in_y},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${data_out}_${data_in} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${data_out}_out        += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
}
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

referenceConvGradB2DTemplate = NodeTemplate("""
// 2D FP ConvGradB NCHW (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${bias}_${grad_out} = ${grad_out};
${bias_type.typeName} ref_${bias}_out = ${bias};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradB2d_fp${grad_out_type.referencedType.typeWidth}_fp${bias_type.referencedType.typeWidth}_NCHW(
        ref_${bias}_${grad_out},
        ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out},
        ref_${bias}_out
    );

    ref_${bias}_${grad_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
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
        ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out},
        ref_${weight}_${data_in},
        ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in},
        ${dim_kernel_y}, ${dim_kernel_x},
        ${stride_y}, ${stride_x},
        ref_${weight}_out,
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
    ref_${weight}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
}
""")

referenceDWConvGradX2DTemplate = NodeTemplate("""
// 2D FP DW ConvTranspose HWC (Name: ${nodeName}, Op: ${nodeOp})
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};
for (uint32_t n=0; n<${batch}; ++n) {
    PULP_DWConvTrans2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${weight},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${data_out}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
}
""")