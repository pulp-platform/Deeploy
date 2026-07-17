# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Adam UpdateV - First Moment Update (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    float32_t ${nodeName}_alpha = ${alpha};
    float32_t ${nodeName}_norm_coef = ${norm_coefficient};
    float32_t ${nodeName}_one_minus_alpha = 1.0f - ${nodeName}_alpha;
    for (uint32_t ${nodeName}_i = 0; ${nodeName}_i < ${size}; ${nodeName}_i++) {
        float32_t ${nodeName}_G_reg = ${nodeName}_norm_coef * ${X}[${nodeName}_i] + ${G}[${nodeName}_i];
        ${V_new}[${nodeName}_i] = ${nodeName}_alpha * ${V}[${nodeName}_i] + ${nodeName}_one_minus_alpha * ${nodeName}_G_reg;
    }
END_SINGLE_CORE
""")
