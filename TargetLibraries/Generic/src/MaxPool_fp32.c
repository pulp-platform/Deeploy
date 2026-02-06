/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void MaxPool2d_fp32_fp32_NCHW(float32_t const *__restrict__ pSrcA, uint32_t C,
                              uint32_t H, uint32_t W, uint32_t P, uint32_t Q,
                              uint32_t SP, uint32_t SQ,
                              float32_t *__restrict__ pDstC) {

  if (H < P || W < Q || SP == 0 || SQ == 0) {
    return;
  }
  uint32_t H_out = (H - P) / SP + 1;
  uint32_t W_out = (W - Q) / SQ + 1;

  for (uint32_t c = 0; c < C; ++c) {
    for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
        float32_t max = -inf;
        for (uint32_t p = 0; p < P; ++p) {
          uint32_t h_in = h_out * SP + p;
          if (h_in >= H)
            continue;

          for (uint32_t q = 0; q < Q; ++q) {
            uint32_t w_in = w_out * SQ + q;
            if (w_in >= W)
              continue;

            float32_t tmp = pSrcA[c * H * W + h_in * W + w_in];
            if (tmp > max) {
              max = tmp;
            }
          }
        }
        pDstC[c * H_out * W_out + h_out * W_out + w_out] = max;
      }
    }
  }
}

void MaxPool1d_fp32_fp32(float32_t const *__restrict__ pSrcA, uint32_t C,
                         uint32_t W, uint32_t K, uint32_t S,
                         float32_t *__restrict__ pDstC) {
  if (W < K || S == 0) {
    return;
  }
  uint32_t W_out = (W - K) / S + 1;
  for (uint32_t c = 0; c < C; ++c) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      float32_t max = -inf;
      for (uint32_t k = 0; k < K; ++k) {
        uint32_t w_in = w_out * S + k;
        if (w_in >= W)
          continue;
        float32_t tmp = pSrcA[c * W + w_in];
        if (tmp > max) {
          max = tmp;
        }
      }
      pDstC[c * W_out + w_out] = max;
    }
  }
}