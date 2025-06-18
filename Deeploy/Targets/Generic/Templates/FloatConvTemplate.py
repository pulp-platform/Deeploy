# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

reference2DTemplate = NodeTemplate("""
<%
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D FP Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        Conv2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_NCHW(
            ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},
            ${weight}, ${ch_im_out}, ${dim_kernel_x}, ${dim_kernel_y},
            ${stride_x}, ${stride_y},
            ${bias},
            ${has_bias},
            ref_${data_out}_${data_out}
        );
        ref_${data_out}_${data_in} += ${batchOffsetIn};
        ref_${data_out}_${data_out} += ${batchOffsetOut};
    }
END_SINGLE_CORE
""")

reference1DTemplate = NodeTemplate("""
    // 1D FP Conv (Name: ${nodeName}, Op: ${nodeOp})
    BEGIN_SINGLE_CORE
        ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
        ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};
        for (uint32_t n=0; n<${batch}; ++n) {
            Conv1d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
                ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_x},
                ${weight}, ${ch_im_out}, ${dim_kernel_x},
                ${stride_x},
                ${bias},
                ${has_bias},
                ref_${data_out}_${data_out},
                ${dim_im_out_y}
            );
        }
                                   
        // Stampa output 
        for (int b = 0; b < ${batch}; ++b) {
            printf("Batch %d:\\n", b);
            for (int c = 0; c < ${ch_im_out}; ++c) {
                printf("Channel %d: ", c);
                for (int x = 0; x < ${dim_im_out_y}; ++x) {
                    int idx = b * (${ch_im_out} * ${dim_im_out_y}) + c * ${dim_im_out_y} + x;
                    printf("%f ", ref_${data_out}_${data_out}[idx]);
                }
                printf("\\n");
            }
        }
    END_SINGLE_CORE
    """)