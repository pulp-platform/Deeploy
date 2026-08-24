# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate2D = NodeTemplate("""
<%
batch_stride_input = channels * input_height * input_width
batch_stride_output = feature_maps * output_height * output_width
%>

// 2D Transposed Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch_size}; ++n) {
        ConvTranspose2d_fp32(
            ref_${data_out}_${data_in}, ${channels}, ${input_height},
            ${input_width}, ${weight}, ${feature_maps}, ${kernel_height},
            ${kernel_width}, ${stride_h}, ${stride_w}, ${bias}, ${has_bias},
            ref_${data_out}_${data_out}, ${output_height}, ${output_width}
        );

        ref_${data_out}_${data_in} += ${batch_stride_input};
        ref_${data_out}_${data_out} += ${batch_stride_output};
    }
END_SINGLE_CORE
""")

referenceTemplate1D = NodeTemplate("""
<%
batch_stride_input = channels * input_length
batch_stride_output = feature_maps * output_length
%>

// 1D Transposed Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch_size}; ++n) {
        ConvTranspose1d_fp32(
            ref_${data_out}_${data_in}, ${channels}, ${input_length}, ${weight},
            ${feature_maps}, ${kernel_length}, ${stride}, ${bias}, ${has_bias},
            ref_${data_out}_${data_out}, ${output_length}
        );

        ref_${data_out}_${data_in} += ${batch_stride_input};
        ref_${data_out}_${data_out} += ${batch_stride_output};
    }
END_SINGLE_CORE
""")
