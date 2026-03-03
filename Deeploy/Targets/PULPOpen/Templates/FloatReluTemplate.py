# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// ReLU (Name: ${nodeName}, Op: ${nodeOp})
PULP_Relu_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
    ${data_in},
    ${data_out},
    ${size}
);
""")

referenceGradTemplate = NodeTemplate("""
// ReLU Grad (Name: ${nodeName}, Op: ${nodeOp})

PULP_ReluGrad_fp${grad_in_type.referencedType.typeWidth}_fp${grad_out_type.referencedType.typeWidth}(
    ${grad_out},
    ${data_in},
    ${grad_in},
    ${size}
);
""")
