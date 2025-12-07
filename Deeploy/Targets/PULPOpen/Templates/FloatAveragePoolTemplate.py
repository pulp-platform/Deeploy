# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// 2D Float AveragePool Channel Parallel (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_AvgPool2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in},
        ${dim_kernel_y}, ${dim_kernel_x},
        ${stride_y}, ${stride_x},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );
    ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")

referenceCHWTemplate = NodeTemplate("""
// 2D Float AveragePool CHW (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_AvgPool2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_CHW(
        ref_${data_out}_${data_in},
        ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in},
        ${dim_kernel_y}, ${dim_kernel_x},
        ${stride_y}, ${stride_x},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );
    ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")

referenceGradTemplate = NodeTemplate("""
// 2D Float AveragePoolGrad Channel Parallel (Name: ${nodeName}, Op: ${nodeOp}) 
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_AvgPoolGrad2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in},
        ${dim_kernel_y}, ${dim_kernel_x},
        ${stride_y}, ${stride_x},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );
    ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")