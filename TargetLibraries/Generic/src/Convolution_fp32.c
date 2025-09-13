/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void Conv2d_fp32_fp32_fp32_NCHW(const float32_t *__restrict__ pSrcA, uint32_t C,
                                uint32_t H_padded, uint32_t W_padded,
                                const float32_t *__restrict__ pSrcB, uint32_t F,
                                uint32_t P, uint32_t Q, uint32_t SP,
                                uint32_t SQ,
                                const float32_t *__restrict__ pSrcBias,
                                const bool has_bias,
                                float32_t *__restrict__ pDstC) {
  // Compute the output dimensions
  uint32_t H_out = (H_padded - P) / SP + 1;
  uint32_t W_out = (W_padded - Q) / SQ + 1;

  // Prepare variables
  uint32_t c, h, w, f, p, q;

  // Compute output with bias
  if (has_bias) {
    for (f = 0; f < F; ++f) {
      for (h = 0; h < H_out; ++h) {
        for (w = 0; w < W_out; ++w) {
          float32_t sum = 0.0f;

          for (c = 0; c < C; ++c) {
            for (p = 0; p < P; ++p) {
              for (q = 0; q < Q; ++q) {
                sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
                             (w * SQ + q)] *
                       pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
              }
            }
          }

          pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
        }
      }
    }
  }
  // Compute output without bias
  else {
    for (f = 0; f < F; ++f) {
      for (h = 0; h < H_out; ++h) {
        for (w = 0; w < W_out; ++w) {
          float32_t sum = 0.0f;

          for (c = 0; c < C; ++c) {
            for (p = 0; p < P; ++p) {
              for (q = 0; q < Q; ++q) {
                sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
                             (w * SQ + q)] *
                       pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
              }
            }
          }

          pDstC[f * H_out * W_out + h * W_out + w] = sum;
        }
      }
    }
  }
}
