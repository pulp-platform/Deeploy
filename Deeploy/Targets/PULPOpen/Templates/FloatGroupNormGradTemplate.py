# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceGradTemplate = NodeTemplate("""
// Float GroupNormGrad (Name: ${nodeName}, Op: ${nodeOp})
// Inputs:  dY[N,C,H,W], X[N,C,H,W], gamma[C], stat[N,G,2]
// Outputs: dX[N,C,H,W] (primary), weight_grad[C], bias_grad[C]

int32_t ${nodeName}_N = ${N};
int32_t ${nodeName}_C = ${C};
int32_t ${nodeName}_H = ${H};
int32_t ${nodeName}_W = ${W};
int32_t ${nodeName}_num_groups = ${num_groups};

// Stack-allocated gradient statistics for this N-tile: [N_tile, G, 2]
float32_t ${nodeName}_grad_stat[${N} * ${num_groups} * 2];

// Step 1 (core 0): compute S1/S2 gradient statistics for this N-tile
if (pi_core_id() == 0) {
    GroupNormGradXStat_fp32_fp32(
        ${dY},
        ${X},
        ${gamma},
        ${stat},
        ${nodeName}_grad_stat,
        ${nodeName}_N,
        ${nodeName}_C,
        ${nodeName}_H,
        ${nodeName}_W,
        ${nodeName}_num_groups
    );
}
pi_cl_team_barrier();

// Step 2 (all cores): compute dX using pre-computed statistics
float32_t *${nodeName}_mean_ptr    = ${stat};
float32_t *${nodeName}_inv_std_ptr = ${stat} + 1;
GroupNormGradX_fp32_fp32(
    ${dY},
    ${X},
    ${gamma},
    ${nodeName}_mean_ptr,
    ${nodeName}_inv_std_ptr,
    ${nodeName}_grad_stat,
    ${dX},
    ${nodeName}_N,
    ${nodeName}_C,
    ${nodeName}_H,
    ${nodeName}_W,
    ${nodeName}_num_groups
);
pi_cl_team_barrier();

// Step 3 (core 0): accumulate weight_grad (dGamma) and bias_grad (dBeta) for this N-tile
static uint8_t ${nodeName}_param_initialized = 0;
if (pi_core_id() == 0) {
    if (!${nodeName}_param_initialized) {
        memset(${weight_grad}, 0, ${nodeName}_C * sizeof(float32_t));
        memset(${bias_grad},   0, ${nodeName}_C * sizeof(float32_t));
        ${nodeName}_param_initialized = 1;
    }
    int32_t ${nodeName}_cpg = ${nodeName}_C / ${nodeName}_num_groups;
    for (int32_t ${nodeName}_n = 0; ${nodeName}_n < ${nodeName}_N; ${nodeName}_n++) {
        for (int32_t ${nodeName}_c = 0; ${nodeName}_c < ${nodeName}_C; ${nodeName}_c++) {
            int32_t ${nodeName}_g        = ${nodeName}_c / ${nodeName}_cpg;
            int32_t ${nodeName}_stat_idx = (${nodeName}_n * ${nodeName}_num_groups + ${nodeName}_g) * 2;
            float32_t ${nodeName}_mu     = ${stat}[${nodeName}_stat_idx];
            float32_t ${nodeName}_isd    = ${stat}[${nodeName}_stat_idx + 1];
            for (int32_t ${nodeName}_h = 0; ${nodeName}_h < ${nodeName}_H; ${nodeName}_h++) {
                for (int32_t ${nodeName}_w = 0; ${nodeName}_w < ${nodeName}_W; ${nodeName}_w++) {
                    int32_t ${nodeName}_idx = ((${nodeName}_n * ${nodeName}_C + ${nodeName}_c) * ${nodeName}_H + ${nodeName}_h) * ${nodeName}_W + ${nodeName}_w;
                    float32_t ${nodeName}_x_norm = (${X}[${nodeName}_idx] - ${nodeName}_mu) * ${nodeName}_isd;
                    ${weight_grad}[${nodeName}_c] += ${dY}[${nodeName}_idx] * ${nodeName}_x_norm;
                    ${bias_grad}[${nodeName}_c]   += ${dY}[${nodeName}_idx];
                }
            }
        }
    }
}
""")
