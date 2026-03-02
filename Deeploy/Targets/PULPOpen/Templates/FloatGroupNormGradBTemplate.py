# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float GroupNormGradB (Name: ${nodeName}, Op: ${nodeOp})
// Computes dBeta = sum(dY) over N, H, W dimensions for each channel

int8_t ${nodeName}_core_id = pi_core_id();

int32_t ${nodeName}_N = ${N};
int32_t ${nodeName}_C = ${C};
int32_t ${nodeName}_H = ${H};
int32_t ${nodeName}_W = ${W};

if(pi_core_id() == 0) {
    // Zero-initialize output tensor on first tile
    static int ${nodeName}_initialized = 0;
    if (!${nodeName}_initialized) {
        memset(${dBeta}, 0, (${nodeName}_C) * 4);
        ${nodeName}_initialized = 1;
    }

    // Sum dY over N, H, W for each channel
    GroupNormGradB_fp32_fp32(
        ${dY},                      // Upstream gradient (tiled)
        ${dBeta},                   // Output gradient for beta [C]
        ${nodeName}_N,              // Batch size
        ${nodeName}_C,              // Number of channels
        ${nodeName}_H,              // Height of this tile
        ${nodeName}_W               // Width of this tile
    );
}
""")
