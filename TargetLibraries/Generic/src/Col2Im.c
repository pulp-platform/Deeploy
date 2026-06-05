// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include "DeeployBasicMath.h"

// clang-format off
#define DEFINE_COL2IM_FN(SUFFIX, DATA_TYPE)                                    \
  DECLARE_COL2IM_FN(SUFFIX, DATA_TYPE) {                                       \
    if (spatial_dims < 1 || spatial_dims > COL2IM_MAX_SPATIAL_DIMS ||          \
        N <= 0 || C <= 0)                                                      \
      return;                                                                  \
    for (int32_t p = 0; p < spatial_dims; p++) {                               \
      if (image_shape[p] <= 0 || block_shape[p] <= 0 || strides[p] <= 0 ||     \
          dilations[p] <= 0 || pads[p] < 0 || pads[p + spatial_dims] < 0)      \
        return;                                                                \
    }                                                                          \
    /* Compute per-dim sliding-window sizes, L, block_volume, image_volume. */ \
    int32_t col_dims[COL2IM_MAX_SPATIAL_DIMS];                                 \
    int32_t col_strides[COL2IM_MAX_SPATIAL_DIMS];                              \
    int32_t blk_strides[COL2IM_MAX_SPATIAL_DIMS];                              \
    int32_t img_strides[COL2IM_MAX_SPATIAL_DIMS];                              \
    int32_t L = 1, block_volume = 1, image_volume = 1;                         \
    for (int32_t p = 0; p < spatial_dims; p++) {                               \
      col_dims[p] = (image_shape[p] + pads[p] + pads[p + spatial_dims]         \
                     - dilations[p] * (block_shape[p] - 1) - 1)                \
                    / strides[p] + 1;                                          \
      if (col_dims[p] <= 0) return;                                            \
      L            *= col_dims[p];                                             \
      block_volume *= block_shape[p];                                          \
      image_volume *= image_shape[p];                                          \
    }                                                                          \
    /* Row-major strides for flat-index decomposition. */                      \
    col_strides[spatial_dims - 1] = 1;                                         \
    blk_strides[spatial_dims - 1] = 1;                                         \
    img_strides[spatial_dims - 1] = 1;                                         \
    for (int32_t p = spatial_dims - 2; p >= 0; p--) {                          \
      col_strides[p] = col_strides[p + 1] * col_dims[p + 1];                   \
      blk_strides[p] = blk_strides[p + 1] * block_shape[p + 1];                \
      img_strides[p] = img_strides[p + 1] * image_shape[p + 1];                \
    }                                                                          \
    /* Zero-initialise output. */                                              \
    memset(output, 0, (size_t)(N * C * image_volume) * sizeof(DATA_TYPE));     \
    /* Accumulate each column entry into its image position. */                \
    for (int32_t n = 0; n < N; n++) {                                          \
      for (int32_t c = 0; c < C; c++) {                                        \
        for (int32_t bk = 0; bk < block_volume; bk++) {                        \
          /* Decompose kernel flat index once per bk. */                       \
          int32_t k_coords[COL2IM_MAX_SPATIAL_DIMS];                           \
          int32_t bk_rem = bk;                                                 \
          for (int32_t p = 0; p < spatial_dims; p++) {                         \
            k_coords[p] = bk_rem / blk_strides[p];                             \
            bk_rem     -= k_coords[p] * blk_strides[p];                        \
          }                                                                    \
          for (int32_t ob = 0; ob < L; ob++) {                                 \
            /* Decompose output-block flat index and map to image coords. */   \
            int32_t ob_rem = ob, img_flat = 0, in_bounds = 1;                  \
            for (int32_t p = 0; p < spatial_dims; p++) {                       \
              int32_t o_coord = ob_rem / col_strides[p];                       \
              ob_rem -= o_coord * col_strides[p];                              \
              int32_t h = o_coord * strides[p] - pads[p]                       \
                          + k_coords[p] * dilations[p];                        \
              if (h < 0 || h >= image_shape[p]) { in_bounds = 0; break; }      \
              img_flat += h * img_strides[p];                                  \
            }                                                                  \
            if (in_bounds) {                                                   \
              int32_t in_flat = (n * C * block_volume                          \
                                 + c * block_volume + bk) * L + ob;            \
              output[(n * C + c) * image_volume + img_flat] += input[in_flat]; \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
// clang-format on

DEFINE_COL2IM_FN(fp32, float32_t)
DEFINE_COL2IM_FN(s8, int8_t)
DEFINE_COL2IM_FN(u8, uint8_t)
