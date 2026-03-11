/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_MaxPool2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                  uint32_t W, uint32_t H, uint32_t C,
                                  uint32_t Q, uint32_t P, uint32_t SQ,
                                  uint32_t SP, float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C);
  uint16_t ch_count = ch_stop - ch_start;

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        float32_t max_val = -inf;

        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;
        for (uint32_t p = 0; p < P; ++p) {
          int32_t h_in = h_in_start + p;

          if (h_in < 0 || h_in >= (int32_t)H) {
            continue;
          }

          for (uint32_t q = 0; q < Q; ++q) {
            int32_t w_in = w_in_start + q;

            if (w_in < 0 || w_in >= (int32_t)W) {
              continue;
            }

            uint32_t input_idx = (h_in * W + w_in) * C + c;
            float32_t val = pSrcA[input_idx];

            if (val > max_val) {
              max_val = val;
            }
          }
        }

        uint32_t output_idx = (h_out * W_out + w_out) * C + c;
        pDstC[output_idx] = max_val;
      }
    }
  }
}

void PULP_MaxPoolGrad2d_fp32_fp32_HWC(const float32_t *__restrict__ pGradOut,
                                      const float32_t *__restrict__ pInput,
                                      uint32_t H_out, uint32_t W_out, uint32_t C,
                                      uint32_t H_in, uint32_t W_in,
                                      uint32_t P, uint32_t Q, uint32_t SP,
                                      uint32_t SQ, float32_t *__restrict__ pGradIn,
                                      uint32_t pad_top, uint32_t pad_bottom,
                                      uint32_t pad_left, uint32_t pad_right) {

  int8_t core_id  = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop  = MIN(ch_start + ch_chunk, C);

  /* Zero-initialise the gradient input for our channel slice */
  for (uint32_t h = 0; h < H_in; ++h) {
    for (uint32_t w = 0; w < W_in; ++w) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        pGradIn[(h * W_in + w) * C + c] = 0.0f;
      }
    }
  }

  /* Scatter upstream gradient to the argmax position in each pooling window */
  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {

      int32_t h_in_start = (int32_t)h_out * (int32_t)SP - (int32_t)pad_top;
      int32_t w_in_start = (int32_t)w_out * (int32_t)SQ - (int32_t)pad_left;

      for (uint32_t c = ch_start; c < ch_stop; ++c) {

        /* Find the argmax position within the pooling window */
        float32_t max_val = -inf;
        int32_t   max_h   = -1;
        int32_t   max_w   = -1;

        for (uint32_t p = 0; p < P; ++p) {
          int32_t h_in = h_in_start + (int32_t)p;
          if (h_in < 0 || h_in >= (int32_t)H_in) continue;

          for (uint32_t q = 0; q < Q; ++q) {
            int32_t w_in = w_in_start + (int32_t)q;
            if (w_in < 0 || w_in >= (int32_t)W_in) continue;

            float32_t val = pInput[((uint32_t)h_in * W_in + (uint32_t)w_in) * C + c];
            if (val > max_val) {
              max_val = val;
              max_h   = h_in;
              max_w   = w_in;
            }
          }
        }

        /* Accumulate upstream gradient at the argmax position */
        if (max_h >= 0 && max_w >= 0) {
          uint32_t out_idx = (h_out * W_out + w_out) * C + c;
          uint32_t in_idx  = ((uint32_t)max_h * W_in + (uint32_t)max_w) * C + c;
          pGradIn[in_idx] += pGradOut[out_idx];
        }
      }
    }
  }
}