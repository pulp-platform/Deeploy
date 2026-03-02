# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float GroupNormGradX with HW Tiling (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);

int32_t ${nodeName}_N = ${N};
int32_t ${nodeName}_C = ${C};
int32_t ${nodeName}_H = ${H};
int32_t ${nodeName}_W = ${W};
int32_t ${nodeName}_num_groups = ${num_groups};


// stat array is interleaved: [mean_0, inv_std_0, mean_1, inv_std_1, ...]
float32_t *${nodeName}_mean_ptr = ${stat};         // Points to mean values (stride 2)
float32_t *${nodeName}_inv_std_ptr = ${stat} + 1;  // Points to inv_std values (stride 2)


GroupNormGradX_fp32_fp32(
    ${dY},                      // Upstream gradient [N, C, H, W]
    ${X},                       // Original input [N, C, H, W]
    ${gamma},                   // Scale parameter [C] (tiled)
    ${nodeName}_mean_ptr,       // Pre-computed mean from forward [N, G] (stride 2)
    ${nodeName}_inv_std_ptr,    // Pre-computed inv_std from forward [N, G] (stride 2)
    ${grad_stat},               // Pre-computed gradient stats [N, G, 2]
    ${dX},                      // Output gradient [N, C, H, W]
    ${nodeName}_N,              // Batch size
    ${nodeName}_C,              // Number of channels (full)
    ${nodeName}_H,              // Height (full)
    ${nodeName}_W,              // Width (full)
    ${nodeName}_num_groups     // Number of groups
);

""")
