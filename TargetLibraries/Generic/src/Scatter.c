/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

// clang-format off
#define DEFINE_SCATTER_FN(SUFFIX, DATA_TYPE)                                   \
  DECLARE_SCATTER_FN(SUFFIX, DATA_TYPE) {                                      \
    int32_t data_size = 1;                                                     \
    for (int32_t dim = 0; dim < ndim; dim++) {                                 \
      data_size *= data_shape[dim];                                            \
    }                                                                          \
    int32_t indices_size = 1;                                                  \
    for (int32_t dim = 0; dim < ndim; dim++) {                                 \
      indices_size *= indices_shape[dim];                                      \
    }                                                                          \
    memcpy(output, data, (size_t)data_size * sizeof(DATA_TYPE));               \
    int32_t stride_data[SCATTER_MAX_NDIM];                                     \
    int32_t stride_idx[SCATTER_MAX_NDIM];                                      \
    stride_data[ndim - 1] = 1;                                                 \
    stride_idx[ndim - 1] = 1;                                                  \
    for (int32_t dim = ndim - 2; dim >= 0; dim--) {                            \
      stride_data[dim] = stride_data[dim + 1] * data_shape[dim + 1];           \
      stride_idx[dim] = stride_idx[dim + 1] * indices_shape[dim + 1];          \
    }                                                                          \
    for (int32_t fi = 0; fi < indices_size; fi++) {                            \
      int32_t out_idx = 0;                                                     \
      int32_t rem = fi;                                                        \
      for (int32_t dim = 0; dim < ndim; dim++) {                               \
        int32_t coord = rem / stride_idx[dim];                                 \
        rem -= coord * stride_idx[dim];                                        \
        if (dim == axis) {                                                     \
          int32_t scatter_idx = indices[fi];                                   \
          if (scatter_idx < 0) scatter_idx += data_shape[dim];                 \
          out_idx += scatter_idx * stride_data[dim];                           \
        } else {                                                               \
          out_idx += coord * stride_data[dim];                                 \
        }                                                                      \
      }                                                                        \
      switch (reduction) {                                                     \
      case SCATTER_REDUCTION_ADD:                                              \
        output[out_idx] += updates[fi];                                        \
        break;                                                                 \
      case SCATTER_REDUCTION_MUL:                                              \
        output[out_idx] *= updates[fi];                                        \
        break;                                                                 \
      case SCATTER_REDUCTION_MIN:                                              \
        if (updates[fi] < output[out_idx])                                     \
          output[out_idx] = updates[fi];                                       \
        break;                                                                 \
      case SCATTER_REDUCTION_MAX:                                              \
        if (updates[fi] > output[out_idx])                                     \
          output[out_idx] = updates[fi];                                       \
        break;                                                                 \
      default:                                                                 \
        output[out_idx] = updates[fi]; break;                                  \
      }                                                                        \
    }                                                                          \
  }
// clang-format on

DEFINE_SCATTER_FN(fp32, float32_t)
DEFINE_SCATTER_FN(s8, int8_t)
DEFINE_SCATTER_FN(u8, uint8_t)