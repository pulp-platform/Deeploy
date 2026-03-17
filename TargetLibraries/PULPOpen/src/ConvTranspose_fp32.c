/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

__attribute__((noinline, optnone)) void PULP_ConvTranspose2d_fp32_fp32_fp32_CHW(
    const float32_t *__restrict__ pSrcA, uint32_t C_in, uint32_t H_in,
    uint32_t W_in, const float32_t *__restrict__ pSrcB, uint32_t C_out,
    uint32_t groups, uint32_t K_h, uint32_t K_w, uint32_t stride_h,
    uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, const float32_t *__restrict__ pSrcBias,
    const bool has_bias, float32_t *__restrict__ pDstC, uint32_t H_out,
    uint32_t W_out) {

  (void)pad_bottom;
  (void)pad_right;

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_out_chunk =
      (C_out >> log2Core) + ((C_out & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, C_out);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, C_out);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  uint32_t output_plane = H_out * W_out;

  for (uint32_t cout = ch_out_start; cout < ch_out_stop; ++cout) {
    float32_t init = has_bias ? pSrcBias[cout] : 0.0f;
    float32_t *out_ptr = pDstC + cout * output_plane;
    for (uint32_t idx = 0; idx < output_plane; ++idx) {
      out_ptr[idx] = init;
    }
  }

  uint32_t ch_in_per_group = C_in / groups;
  uint32_t ch_out_per_group = C_out / groups;

  for (uint32_t cout = ch_out_start; cout < ch_out_stop; ++cout) {
    uint32_t group_idx = cout / ch_out_per_group;
    uint32_t cout_in_group = cout % ch_out_per_group;

    for (uint32_t cin_in_group = 0; cin_in_group < ch_in_per_group;
         ++cin_in_group) {
      uint32_t cin = group_idx * ch_in_per_group + cin_in_group;

      for (uint32_t h_in = 0; h_in < H_in; ++h_in) {
        for (uint32_t w_in = 0; w_in < W_in; ++w_in) {
          float32_t val = pSrcA[(cin * H_in + h_in) * W_in + w_in];

          for (uint32_t kh = 0; kh < K_h; ++kh) {
            int32_t h_out =
                (int32_t)(h_in * stride_h + kh * dilation_h) - (int32_t)pad_top;

            if (h_out < 0 || h_out >= (int32_t)H_out) {
              continue;
            }

            for (uint32_t kw = 0; kw < K_w; ++kw) {
              int32_t w_out = (int32_t)(w_in * stride_w + kw * dilation_w) -
                              (int32_t)pad_left;

              if (w_out < 0 || w_out >= (int32_t)W_out) {
                continue;
              }

              uint32_t weight_idx =
                  (((cin * ch_out_per_group + cout_in_group) * K_h + kh) *
                   K_w) +
                  kw;
              uint32_t out_idx =
                  (cout * H_out + (uint32_t)h_out) * W_out + (uint32_t)w_out;

              pDstC[out_idx] += val * pSrcB[weight_idx];
            }
          }
        }
      }
    }
  }
}
