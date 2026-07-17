# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Adam UpdateH - Second Moment Update (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    float32_t ${nodeName}_beta_coeff = ${beta};
    float32_t ${nodeName}_norm_coef = ${norm_coefficient};
    float32_t ${nodeName}_one_minus_beta = 1.0f - ${nodeName}_beta_coeff;
    for (uint32_t ${nodeName}_i = 0; ${nodeName}_i < ${size}; ${nodeName}_i++) {
        float32_t ${nodeName}_G_reg = ${nodeName}_norm_coef * ${X}[${nodeName}_i] + ${G}[${nodeName}_i];
        ${H_new}[${nodeName}_i] = ${nodeName}_beta_coeff * ${H}[${nodeName}_i] + ${nodeName}_one_minus_beta * ${nodeName}_G_reg * ${nodeName}_G_reg;
    }
END_SINGLE_CORE
""")
