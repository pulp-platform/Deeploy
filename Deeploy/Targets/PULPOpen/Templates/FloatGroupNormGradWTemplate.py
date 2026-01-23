# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float GroupNormGradW with HW Tiling (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);

int32_t ${nodeName}_N = ${N};
int32_t ${nodeName}_C = ${C};
int32_t ${nodeName}_H = ${H};  // Tiled H from TileConstraint
int32_t ${nodeName}_W = ${W};  // Tiled W from TileConstraint
int32_t ${nodeName}_num_groups = ${num_groups};
// stat array is interleaved: [mean_0, inv_std_0, mean_1, inv_std_1, ...]
// Pass pointers to stat[0] and stat[1] for stride-2 access
float32_t *${nodeName}_mean_ptr = ${stat};         // Points to mean values (stride 2)
float32_t *${nodeName}_inv_std_ptr = ${stat} + 1;  // Points to inv_std values (stride 2)
float32_t ${nodeName}_epsilon = 0.0f;

if(pi_core_id() == 0) {
// Zero-initialize output tensor on first tile
static int ${nodeName}_initialized = 0;
if (!${nodeName}_initialized) {
  memset(${dGamma}, 0, (${nodeName}_C) * 4);
  ${nodeName}_initialized = 1;
}

// Each core processes a subset of channels
    GroupNormGradW_fp32_fp32(
        ${dY},                      // Upstream gradient (tiled)
        ${X},                       // Original input (tiled)
        ${nodeName}_mean_ptr,       // Pre-computed mean [N, G] (stride 2 in stat array)
        ${nodeName}_inv_std_ptr,    // Pre-computed inv_std [N, G] (stride 2 in stat array)
        ${dGamma},                  // Output gradient for gamma [C]
        ${nodeName}_N,              // Batch size
        ${nodeName}_C,              // Number of channels
        ${nodeName}_H,              // Height of this tile
        ${nodeName}_W,              // Width of this tile
        ${nodeName}_num_groups,     // Number of groups
        ${nodeName}_epsilon,        // Epsilon
        0,                          // Start channel index
        ${nodeName}_C               // End channel index
    );

}
""")
