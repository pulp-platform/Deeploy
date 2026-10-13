# SPDX-FileCopyrightText: 2026 EPFL
#
# SPDX-License-Identifier: Apache-2.0
#
# Author: Mohammad Hossein Nikkhah

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// ReduceMax (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    uint32_t outer_base = 0;
    uint32_t inner_index = 0;
    uint32_t input_base;


    for (uint32_t i=0;i<${output_size};i++){
        input_base = outer_base + inner_index;
        uint32_t input_offset = input_base;


        ${data_in_type.referencedType.typeName} max_value = ${data_in}[input_offset];



        for (uint32_t i_a = 0; i_a < ${d_axes}; i_a++) {
            // Max operation :
            if (max_value <  ${data_in}[input_offset])
                max_value = ${data_in}[input_offset];

            input_offset += ${inner_size};
        }

        ${data_out}[i] = max_value;

        inner_index ++;

        if (inner_index >= ${inner_size}) {
            inner_index = 0;
            outer_base += ${outer_step};
        }

    }
END_SINGLE_CORE
""")