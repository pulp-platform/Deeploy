# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

reference1DTemplate = NodeTemplate("""
<%
bias_ptr = bias if 'bias' in locals() else "NULL"
batchOffsetIn = ch_im_in * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_y
%>

// 1D Transposed Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        ConvTranspose1d_fp32(
            ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_y},
            ${weight}, ${ch_im_out}, ${dim_kernel_y},
            ${stride_y},
            ${bias_ptr}, ${has_bias},
            ref_${data_out}_${data_out}, ${dim_im_out_y}
        );

        ref_${data_out}_${data_in} += ${batchOffsetIn};
        ref_${data_out}_${data_out} += ${batchOffsetOut};
    }
END_SINGLE_CORE
""")

reference2DTemplate = NodeTemplate("""
<%
bias_ptr = bias if 'bias' in locals() else "NULL"
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D Transposed Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        ConvTranspose2d_fp32(
            ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},
            ${weight}, ${ch_im_out}, ${dim_kernel_x}, ${dim_kernel_y},
            ${stride_x}, ${stride_y},
            ${bias_ptr}, ${has_bias},
            ref_${data_out}_${data_out}, ${dim_im_out_x}, ${dim_im_out_y}
        );

        ref_${data_out}_${data_in} += ${batchOffsetIn};
        ref_${data_out}_${data_out} += ${batchOffsetOut};
    }
END_SINGLE_CORE
""")
