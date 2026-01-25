/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

void GroupNormGradXStat_fp32_fp32(const float32_t *dY,
                                  const float32_t *X,
                                  const float32_t *gamma,
                                  const float32_t *stat,
                                  float32_t *grad_stat,
                                  int32_t N, int32_t C, int32_t H, int32_t W,
                                  int32_t num_groups) {
    const int32_t HW = H * W;
    const int32_t Cg = C / num_groups;
    const int32_t group_size = Cg * HW;

    for (int32_t n = 0; n < N; n++) {
        for (int32_t g = 0; g < num_groups; g++) {
            int32_t stat_idx = (n * num_groups + g) * 2;
            float32_t mu = stat[stat_idx];
            float32_t istd = stat[stat_idx + 1];

            const int32_t c_start = g * Cg;
            const int32_t c_end = c_start + Cg;

            float32_t S1 = 0.0f;
            float32_t S2 = 0.0f;

            for (int32_t c = c_start; c < c_end; c++) {
                float32_t gam = gamma[c];
                for (int32_t h = 0; h < H; h++) {
                    for (int32_t w = 0; w < W; w++) {
                        int32_t idx = ((n * C + c) * H + h) * W + w;
                        float32_t dy_g = dY[idx] * gam;
                        float32_t x_norm = (X[idx] - mu) * istd;

                        S1 += dy_g;
                        S2 += dy_g * x_norm;
                    }
                }
            }

            grad_stat[stat_idx] = S1 / (float32_t)group_size;
            grad_stat[stat_idx + 1] = S2 / (float32_t)group_size;
        }
    }
}

/**
 * @brief Apply GroupNormGradX using pre-computed statistics.
 *
 * Uses pre-computed mean_gamma_dY and mean_gamma_dY_Xnorm from grad_stat.
 *
 * @param dY           Upstream gradient [N, C, H, W]
 * @param X            Original input [N, C, H, W]
 * @param gamma        Scale parameter [C]
 * @param mean         Pre-computed mean from forward [N, num_groups] (stride 2)
 * @param inv_std      Pre-computed inv_std from forward [N, num_groups] (stride 2)
 * @param grad_stat    Pre-computed gradient stats [N, num_groups, 2]
 * @param dX           Output gradient [N, C, H, W]
 * @param N            Batch size
 * @param C            Number of channels
 * @param H            Height
 * @param W            Width
 * @param num_groups   Number of groups
 */
void GroupNormGradX_fp32_fp32(const float32_t *dY, const float32_t *X,
                               const float32_t *gamma, const float32_t *mean,
                               const float32_t *inv_std, const float32_t *grad_stat,
                               float32_t *dX,
                               int32_t N, int32_t C, int32_t H, int32_t W,
                               int32_t num_groups) {
    int32_t channels_per_group = C / num_groups;

    // Process all channels
    for (int32_t c = 0; c < C; c++) {
        int32_t group = c / channels_per_group;
        float32_t gamma_c = gamma[c];

        // Process all batch elements
        for (int32_t n = 0; n < N; n++) {
            // Get pre-computed statistics for this group
            int32_t stat_idx = (n * num_groups + group) * 2;
            float32_t group_mean = mean[stat_idx];
            float32_t group_inv_std = inv_std[stat_idx];
            float32_t mean_gamma_dY = grad_stat[stat_idx];
            float32_t mean_gamma_dY_Xnorm = grad_stat[stat_idx + 1];

            // Process all spatial positions
            for (int32_t h = 0; h < H; h++) {
                for (int32_t w = 0; w < W; w++) {
                    int32_t idx = ((n * C + c) * H + h) * W + w;

                    float32_t x_val = X[idx];
                    float32_t dy_val = dY[idx];

                    // Normalize x
                    float32_t x_norm = (x_val - group_mean) * group_inv_std;

                    // Compute gradient
                    // dX = inv_std * (gamma * dY - mean(gamma * dY) - X_norm * mean(gamma * dY * X_norm))
                    float32_t gamma_dy = gamma_c * dy_val;
                    dX[idx] = group_inv_std * (gamma_dy - mean_gamma_dY - x_norm * mean_gamma_dY_Xnorm);
                }
            }
        }
    }
}

void GroupNormGradW_fp32_fp32(const float32_t *dY, const float32_t *X,
                               const float32_t *mean, const float32_t *inv_std,
                               float32_t *dGamma,
                               int32_t N, int32_t C, int32_t H, int32_t W,
                               int32_t num_groups, float32_t epsilon,
                               int32_t start_c, int32_t end_c) {
    int32_t channels_per_group = C / num_groups;

    // Process assigned channels
    for (int32_t c = start_c; c < end_c; c++) {
        // Determine which group this channel belongs to
        int32_t group = c / channels_per_group;

        // Use Kahan summation for better precision
        float32_t sum_dY_Xnorm = 0.0f;
        float32_t sum_c = 0.0f;  // Kahan compensation

        // Sum over all batch and spatial dimensions
        for (int32_t n = 0; n < N; n++) {
            // Get pre-computed statistics for this group (stride 2 for interleaved stat array)
            float32_t group_mean = mean[(n * num_groups + group) * 2];
            float32_t group_inv_std = inv_std[(n * num_groups + group) * 2];

            for (int32_t h = 0; h < H; h++) {
                for (int32_t w = 0; w < W; w++) {
                    int32_t idx = ((n * C + c) * H + h) * W + w;
                    float32_t x_val = X[idx];
                    float32_t dy_val = dY[idx];

                    // Normalize x
                    float32_t x_norm = (x_val - group_mean) * group_inv_std;

                    // Accumulate dGamma with Kahan summation
                    float32_t val = dy_val * x_norm;
                    float32_t y = val - sum_c;
                    float32_t t = sum_dY_Xnorm + y;
                    sum_c = (t - sum_dY_Xnorm) - y;
                    sum_dY_Xnorm = t;
                }
            }
        }

        dGamma[c] += sum_dY_Xnorm;
    }
}

void GroupNormalizationStat_fp32(const float32_t *X, float32_t *stat,
                                  int32_t N, int32_t C, int32_t H, int32_t W,
                                  int32_t num_groups, float32_t epsilon,
                                  int32_t start_group, int32_t end_group) {
    int32_t channels_per_group = C / num_groups;
    int32_t spatial_size = H * W;
    int32_t group_size = channels_per_group * spatial_size;
    printf("epsilon is %f", epsilon );

    // Process assigned groups
    for (int32_t g_idx = start_group; g_idx < end_group; g_idx++) {
        // Convert flat group index to (n, group) coordinates
        int32_t n = g_idx / num_groups;
        int32_t group = g_idx % num_groups;

        // Compute mean for this group using Kahan summation
        float32_t sum = 0.0f;
        float32_t sum_c = 0.0f;  // Kahan compensation
        for (int32_t c_local = 0; c_local < channels_per_group; c_local++) {
            int32_t c = group * channels_per_group + c_local;
            for (int32_t h = 0; h < H; h++) {
                for (int32_t w = 0; w < W; w++) {
                    int32_t idx = ((n * C + c) * H + h) * W + w;
                    float32_t y = X[idx] - sum_c;
                    float32_t t = sum + y;
                    sum_c = (t - sum) - y;
                    sum = t;
                }
            }
        }
        float32_t mean = sum / (float32_t)group_size;

        // Compute variance for this group using Kahan summation
        float32_t var_sum = 0.0f;
        float32_t var_c = 0.0f;  // Kahan compensation
        for (int32_t c_local = 0; c_local < channels_per_group; c_local++) {
            int32_t c = group * channels_per_group + c_local;
            for (int32_t h = 0; h < H; h++) {
                for (int32_t w = 0; w < W; w++) {
                    int32_t idx = ((n * C + c) * H + h) * W + w;
                    float32_t diff = X[idx] - mean;
                    float32_t val = diff * diff;
                    float32_t y = val - var_c;
                    float32_t t = var_sum + y;
                    var_c = (t - var_sum) - y;
                    var_sum = t;
                }
            }
        }
        float32_t variance = var_sum / (float32_t)group_size;

        // Compute inverse std
        float32_t inv_std = 1.0f / sqrtf(variance + epsilon);

        // Store statistics: stat[n * num_groups + group, 0] = mean, stat[n * num_groups + group, 1] = inv_std
        int32_t stat_idx = (n * num_groups + group) * 2;
        stat[stat_idx] = mean;
        stat[stat_idx + 1] = inv_std;
    }
}

void GroupNormalization_fp32(const float32_t *X, const float32_t *gamma,
                              const float32_t *beta, const float32_t *stat,
                              float32_t *Y,
                              int32_t N, int32_t C, int32_t H, int32_t W,
                              int32_t num_groups,
                              int32_t start_group, int32_t end_group) {
    int32_t channels_per_group = C / num_groups;

    // Process assigned groups
    for (int32_t g_idx = start_group; g_idx < end_group; g_idx++) {
        // Convert flat group index to (n, group) coordinates
        int32_t n = g_idx / num_groups;
        int32_t group = g_idx % num_groups;

        // Get pre-computed statistics for this group
        int32_t stat_idx = (n * num_groups + group) * 2;
        float32_t mean = stat[stat_idx];
        float32_t inv_std = stat[stat_idx + 1];

        // Apply normalization: Y = gamma * (X - mean) * inv_std + beta
        for (int32_t c_local = 0; c_local < channels_per_group; c_local++) {
            int32_t c = group * channels_per_group + c_local;
            float32_t gamma_c = gamma[c];
            float32_t beta_c = beta[c];

            for (int32_t h = 0; h < H; h++) {
                for (int32_t w = 0; w < W; w++) {
                    int32_t idx = ((n * C + c) * H + h) * W + w;
                    float32_t x_val = X[idx];
                    float32_t x_norm = (x_val - mean) * inv_std;
                    Y[idx] = gamma_c * x_norm + beta_c;
                }
            }
        }
    }
}

void GroupNormGradB_fp32_fp32(const float32_t *dY, float32_t *dBeta,
                               int32_t N, int32_t C, int32_t H, int32_t W) {
    // dBeta[c] = sum over (n, h, w) of dY[n, c, h, w]
    for (int32_t c = 0; c < C; c++) {
        float32_t sum = 0.0f;
        for (int32_t n = 0; n < N; n++) {
            for (int32_t h = 0; h < H; h++) {
                for (int32_t w = 0; w < W; w++) {
                    int32_t idx = ((n * C + c) * H + h) * W + w;
                    sum += dY[idx];
                }
            }
        }
        dBeta[c] += sum;  // += for tiling accumulation
    }
}
