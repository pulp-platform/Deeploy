# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float GroupNormalizationStat (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);

int32_t ${nodeName}_N = ${N};
int32_t ${nodeName}_C = ${C};
int32_t ${nodeName}_H = ${H};
int32_t ${nodeName}_W = ${W};
int32_t ${nodeName}_num_groups = ${num_groups};

// Each core processes a subset of N * num_groups groups
int32_t ${nodeName}_total_groups = ${nodeName}_N * ${nodeName}_num_groups;
int32_t ${nodeName}_chunk = (${nodeName}_total_groups >> ${nodeName}_log2Core) +
                            ((${nodeName}_total_groups & (NUM_CORES-1)) != 0);
int32_t ${nodeName}_start = MIN(${nodeName}_chunk * ${nodeName}_core_id, ${nodeName}_total_groups);
int32_t ${nodeName}_end = MIN(${nodeName}_start + ${nodeName}_chunk, ${nodeName}_total_groups);

if (${nodeName}_start < ${nodeName}_end) {
    GroupNormalizationStat_fp32(
        ${X},                       // Input tensor [N, C, H, W]
        ${stat},                    // Output stat [N, G, 2] (mean at [:,:,0], inv_std at [:,:,1])
        ${nodeName}_N,              // Batch size
        ${nodeName}_C,              // Number of channels
        ${nodeName}_H,              // Height
        ${nodeName}_W,              // Width
        ${nodeName}_num_groups,     // Number of groups
        ${epsilon},                 // Epsilon
        ${nodeName}_start,          // Start group index
        ${nodeName}_end             // End group index
    );
}
""")
