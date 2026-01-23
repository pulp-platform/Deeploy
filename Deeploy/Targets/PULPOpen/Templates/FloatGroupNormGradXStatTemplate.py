# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float GroupNormGradXStat (Name: ${nodeName}, Op: ${nodeOp})

int32_t ${nodeName}_N = ${N};
int32_t ${nodeName}_C = ${C};
int32_t ${nodeName}_H = ${H};
int32_t ${nodeName}_W = ${W};
int32_t ${nodeName}_num_groups = ${num_groups};


GroupNormGradXStat_fp32_fp32(
    ${dY},                      // Upstream gradient [N, C, H, W]
    ${X},                       // Original input [N, C, H, W]
    ${gamma},                   // Scale parameter [C]
    ${stat},                    // Pre-computed stat from forward [N, G, 2] (stride 2)
    ${grad_stat},               // Output gradient statistics [N, G, 2]
    ${nodeName}_N,              // Batch size
    ${nodeName}_C,              // Number of channels
    ${nodeName}_H,              // Height
    ${nodeName}_W,              // Width
    ${nodeName}_num_groups     // Number of groups
);

""")
