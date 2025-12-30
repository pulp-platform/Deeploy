/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void MaxPool2d_s8_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                          uint32_t H, uint32_t W, uint32_t P, uint32_t Q,
                          uint32_t SP, uint32_t SQ, int8_t *__restrict__ pDstC,
                          int32_t input_offset, int32_t output_offset) {
  // WIESEP: For now assume padding=0
  uint32_t H_out = (H - P) / SP + 1;
  uint32_t W_out = (W - Q) / SQ + 1;

  uint32_t c = 0; // input channel loop counter
  uint32_t h = 0; // input row loop counter
  uint32_t w = 0; // input column loop counter

  uint32_t p = 0; // kernel row loop counter
  uint32_t q = 0; // kernel column loop counter

  int32_t max;
  int32_t volatile tmp;
  for (c = 0; c < C; ++c) {
    for (h = 0; h < H_out; ++h) {
      for (w = 0; w < W_out; ++w) {
        max = -128;
        // printf("(%2d,%2d,%2d) ", c, h, w);
        for (p = 0; p < P; ++p) {
          for (q = 0; q < Q; ++q) {
            tmp = (int32_t)(pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
                            input_offset);
            if (tmp > max) {
              // printf("%4d >  %4d, ", tmp, max);
              max = tmp;
            }
            // else {
            // printf("%4d <= %-4d, ", tmp, max);
            // }
          }
        }
        // printf(" -> %d\r\n", max);
        pDstC[c * H_out * W_out + h * W_out + w] =
            (int8_t)(max + output_offset);
      }
    }
  }
}

void MaxPool1d_s8_s8(int8_t const *__restrict__ pSrcA, uint32_t C, uint32_t L,
                     uint32_t K, uint32_t S, int8_t *__restrict__ pDstC,
                     int32_t input_offset, int32_t output_offset) {
  uint32_t L_out = (L - K) / S + 1;
  for (uint32_t c = 0; c < C; ++c) {
    for (uint32_t l_out = 0; l_out < L_out; ++l_out) {
      int32_t max = -128;
      for (uint32_t k = 0; k < K; ++k) {
        uint32_t l_in = l_out * S + k;
        if (l_in >= L)
          continue;
        int32_t tmp = (int32_t)(pSrcA[c * L + l_in] + input_offset);
        if (tmp > max) {
          max = tmp;
        }
      }
      pDstC[c * L_out + l_out] = (int8_t)(max + output_offset);
    }
  }
}
