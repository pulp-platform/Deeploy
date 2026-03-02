# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""

// Float GlobalAveragePool (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        for (uint32_t c=0; c<${channels}; ++c) {
            ${data_out_type.referencedType.typeName} sum = 0.0;
            for (uint32_t h=0; h<${dim_im_in_x}; ++h) {
                for (uint32_t w=0; w<${dim_im_in_y}; ++w) {
                    uint32_t idx = ((n * ${channels} + c) * ${dim_im_in_x} + h) * ${dim_im_in_y} + w;
                    sum += ref_${data_out}_${data_in}[idx];
                }
            }
            ref_${data_out}_${data_out}[n * ${channels} + c] = sum / ${size};
        }
    }
END_SINGLE_CORE
""")

reference1DTemplate = NodeTemplate("""

// 1D Float GlobalAveragePool (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        for (uint32_t c=0; c<${channels}; ++c) {
            ${data_out_type.referencedType.typeName} sum = 0.0;
            for (uint32_t l=0; l<${dim_im_in_y}; ++l) {
                uint32_t idx = (n * ${channels} + c) * ${dim_im_in_y} + l;
                sum += ref_${data_out}_${data_in}[idx];
            }
            ref_${data_out}_${data_out}[n * ${channels} + c] = sum / ${size};
        }
    }
END_SINGLE_CORE
""")
