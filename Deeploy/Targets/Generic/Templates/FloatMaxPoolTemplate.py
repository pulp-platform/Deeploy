# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""

// 2D Float MaxPool (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        MaxPool2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_NCHW(
            ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},${dim_kernel_x}, ${dim_kernel_y}, ${stride_x}, ${stride_y},
            ref_${data_out}_${data_out}
        );

    }
END_SINGLE_CORE
""")
