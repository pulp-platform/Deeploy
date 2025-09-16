# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// SGD Weight Update (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${weight_type.typeName} ref_${weight} = ${weight};
    ${grad_type.typeName} ref_${grad} = ${grad};
    ${weight_type.typeName} ref_${weight_updated} = ${weight_updated};

    float32_t learning_rate = ${lr};

    for (uint32_t i=0; i<${size}; ++i) {
        ref_${weight_updated}[i] = ref_${weight}[i] - learning_rate * ref_${grad}[i];
    }
END_SINGLE_CORE
""")
