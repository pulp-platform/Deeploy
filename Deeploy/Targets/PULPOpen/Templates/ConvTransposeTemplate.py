# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

reference2DTemplate = NodeTemplate("""
<%
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D FP ConvTranspose CHW on PULPOpen/Siracusa (Name: ${nodeName}, Op: ${nodeOp})
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvTranspose2d_fp32_fp32_fp32_CHW(
        ref_${data_out}_${data_in},
        ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},
        ${weight},
        ${ch_im_out}, ${group},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ${dilation_x}, ${dilation_y},
        ${padding_y_top}, ${padding_y_bottom},
        ${padding_x_left}, ${padding_x_right},
        ${bias}, ${has_bias},
        ref_${data_out}_${data_out},
        ${dim_im_out_x}, ${dim_im_out_y}
    );

    ref_${data_out}_${data_in} += ${batchOffsetIn};
    ref_${data_out}_${data_out} += ${batchOffsetOut};
}
""")
