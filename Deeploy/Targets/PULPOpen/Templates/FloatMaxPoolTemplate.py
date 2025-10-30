# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// 2D Float MaxPool Channel Parallel (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${nodeName}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${nodeName}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_MaxPool2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${nodeName}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${nodeName}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );
    ref_${nodeName}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${nodeName}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")