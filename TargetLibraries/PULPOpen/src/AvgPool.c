/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"


void PULP_AvgPool2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                  uint32_t W, uint32_t H, uint32_t C,
                                  uint32_t Q, uint32_t P, uint32_t SQ,
                                  uint32_t SP, float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right) {
  
  int8_t core_id  = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop  = MIN(ch_start + ch_chunk, C);

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        
        float32_t sum = 0.0f;

        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        for (uint32_t p = 0; p < P; ++p) {
          int32_t h_in = h_in_start + p;

          //RW: Compiler Bug related to continue
          // if (h_in < 0 || h_in >= (int32_t)H) continue;

          for (uint32_t q = 0; q < Q; ++q) {
            int32_t w_in = w_in_start + q;
           // if (w_in < 0 || w_in >= (int32_t)W) continue;

            uint32_t input_idx = (h_in * W + w_in) * C + c;
            sum += pSrcA[input_idx];
          }
        }

        uint32_t output_idx = (h_out * W_out + w_out) * C + c;
        pDstC[output_idx] = sum / (float32_t)(P * Q);
      }
    }
  }
}

void PULP_AvgPool2d_fp32_fp32_CHW(const float32_t *__restrict__ pSrcA,
                                  uint32_t C, uint32_t H, uint32_t W,
                                  uint32_t P, uint32_t Q, uint32_t SP,
                                  uint32_t SQ, float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right)
{
  int8_t core_id  = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop  = MIN(ch_start + ch_chunk, C);

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t c = ch_start; c < ch_stop; ++c) {
    for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      for (uint32_t w_out = 0; w_out < W_out; ++w_out) {

        float32_t sum   = 0.0f;

        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        for (uint32_t p = 0; p < P; ++p) {
          int32_t h_in = h_in_start + (int32_t)p;
  
          for (uint32_t q = 0; q < Q; ++q) {
            int32_t sw_in = w_in_start + (int32_t)q;
        
            uint32_t input_idx = c * (H * W) + h_in * W + sw_in;
            sum   += pSrcA[input_idx];

        }
      }
        uint32_t output_idx = c * (H_out * W_out) + h_out * W_out + w_out;

        pDstC[output_idx] = sum / (P*Q); 
        printf("output is %f\n", pDstC[output_idx]);
      }
    }
  }
}


void PULP_AvgPoolGrad2d_fp32_fp32_HWC(const float32_t *__restrict__ pGradOut,
                                      uint32_t W_out, uint32_t H_out, uint32_t C,
                                      uint32_t Q, uint32_t P, uint32_t SQ,
                                      uint32_t SP, float32_t *__restrict__ pGradIn,
                                      uint32_t pad_top, uint32_t pad_bottom,
                                      uint32_t pad_left, uint32_t pad_right)
{
  int8_t core_id  = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop  = MIN(ch_start + ch_chunk, C);

  uint32_t H_in = (H_out - 1) * SP + P - pad_top  - pad_bottom;
  uint32_t W_in = (W_out - 1) * SQ + Q - pad_left - pad_right;

  for (uint32_t h = 0; h < H_in; ++h) {
    for (uint32_t w = 0; w < W_in; ++w) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        uint32_t idx = (h * W_in + w) * C + c;
        pGradIn[idx] = 0.0f;
      }
    }
  }


  uint32_t H_pad = H_in + pad_top  + pad_bottom;
  uint32_t W_pad = W_in + pad_left + pad_right;


  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {

      
      int32_t h_start_pad = (int32_t)h_out * (int32_t)SP;
      int32_t w_start_pad = (int32_t)w_out * (int32_t)SQ;

      int32_t h_end_pad = h_start_pad + (int32_t)P;
      int32_t w_end_pad = w_start_pad + (int32_t)Q;

      
      if (h_end_pad > (int32_t)H_pad) h_end_pad = (int32_t)H_pad;
      if (w_end_pad > (int32_t)W_pad) w_end_pad = (int32_t)W_pad;

     
      int32_t actual_h_start = h_start_pad - (int32_t)pad_top;
      int32_t actual_w_start = w_start_pad - (int32_t)pad_left;
      int32_t actual_h_end   = h_end_pad   - (int32_t)pad_top;
      int32_t actual_w_end   = w_end_pad   - (int32_t)pad_left;

      if (actual_h_start < 0)            actual_h_start = 0;
      if (actual_w_start < 0)            actual_w_start = 0;
      if (actual_h_end   > (int32_t)H_in) actual_h_end   = (int32_t)H_in;
      if (actual_w_end   > (int32_t)W_in) actual_w_end   = (int32_t)W_in;

      int32_t win_h = actual_h_end - actual_h_start;
      int32_t win_w = actual_w_end - actual_w_start;

      uint32_t pool_size;
      if (win_h <= 0 || win_w <= 0) {
       
        pool_size = 1;
      } else {
        pool_size = (uint32_t)(win_h * win_w);
      }

      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        uint32_t out_idx = (h_out * W_out + w_out) * C + c;
        float32_t g = pGradOut[out_idx];

        float32_t grad_per_pos = g / (float32_t)pool_size;

        
        for (uint32_t p = 0; p < P; ++p) {
          for (uint32_t q = 0; q < Q; ++q) {
            int32_t in_h = h_start_pad + (int32_t)p - (int32_t)pad_top;
            int32_t in_w = w_start_pad + (int32_t)q - (int32_t)pad_left;

            if ((uint32_t)in_h < H_in && (uint32_t)in_w < W_in) {
              uint32_t in_idx = ((uint32_t)in_h * W_in + (uint32_t)in_w) * C + c;
              pGradIn[in_idx] += grad_per_pos;
            }
          }
        }
      }
    }
  }
}