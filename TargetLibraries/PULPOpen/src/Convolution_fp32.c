/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_Conv2d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right) {

  // Compute core
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  // Compute the chunk size for each core
  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  // Pointer to the weights for the current core
  const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  // Compute the output dimensions
  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  // Compute the output
  if (has_bias) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        for (uint32_t f = 0; f < ch_out_count; ++f) {
          float32_t sum = 0.0f;

          for (uint32_t p = 0; p < P; ++p) {
            for (uint32_t q = 0; q < Q; ++q) {
              for (uint32_t c = 0; c < C; ++c) {
                int32_t h_in = h * SP + p - pad_top;
                int32_t w_in = w * SQ + q - pad_left;

                if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
                    w_in >= (int32_t)W) {
                  continue;
                }

                uint32_t input_idx = (h_in * W + w_in) * C + c;
                uint32_t weight_idx = f * (P * Q * C) + p * (Q * C) + q * C + c;

                sum += pSrcA[input_idx] * weight_ptr[weight_idx];
              }
            }
          }

          uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
          pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
        }
      }
    }
  } else {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        for (uint32_t f = 0; f < ch_out_count; ++f) {
          float32_t sum = 0.0f;

          for (uint32_t p = 0; p < P; ++p) {
            for (uint32_t q = 0; q < Q; ++q) {
              for (uint32_t c = 0; c < C; ++c) {
                int32_t h_in = h * SP + p - pad_top;
                int32_t w_in = w * SQ + q - pad_left;

                if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
                    w_in >= (int32_t)W) {
                  continue;
                }

                uint32_t input_idx = (h_in * W + w_in) * C + c;
                uint32_t weight_idx = f * (P * Q * C) + p * (Q * C) + q * C + c;

                sum += pSrcA[input_idx] * weight_ptr[weight_idx];
              }
            }
          }

          uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
          pDstC[output_idx] = sum;
        }
      }
    }
  }
}

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer) {

  // Compute core
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  // Compute the chunk size for each core
  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  // Pointer to the weights for the current core
  const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  uint32_t im2col_size_per_core = C * P * Q;
  float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

  // Compute the output dimensions
  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  uint32_t kernel_size = P * Q * C;

  // Compute the output
  if (has_bias) {
    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        for (uint32_t p = 0; p < P; p++) {
          int32_t h_in = h_in_start + p;

          for (uint32_t q = 0; q < Q; q++) {
            int32_t w_in = w_in_start + q;

            for (uint32_t c = 0; c < C; c++) {
              if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                  w_in < (int32_t)W) {
                uint32_t in_idx = (h_in * W + w_in) * C + c;
                im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
              } else {
                im2col_buffer[p * Q * C + q * C + c] = 0.0f;
              }
            }
          }
        }

        for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
          float32_t sum = 0.0f;
          const float32_t *local_weight_ptr =
              weight_ptr + (f - ch_out_start) * kernel_size;

          for (uint32_t k = 0; k < kernel_size; k++) {
            sum += im2col_buffer[k] * local_weight_ptr[k];
          }

          uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;

          pDstC[out_idx] = sum + pSrcBias[f];
        }
      }
    }
  } else {
    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        for (uint32_t p = 0; p < P; p++) {
          int32_t h_in = h_in_start + p;

          for (uint32_t q = 0; q < Q; q++) {
            int32_t w_in = w_in_start + q;

            for (uint32_t c = 0; c < C; c++) {
              if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                  w_in < (int32_t)W) {
                uint32_t in_idx = (h_in * W + w_in) * C + c;
                im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
              } else {
                im2col_buffer[p * Q * C + q * C + c] = 0.0f;
              }
            }
          }
        }

        for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
          float32_t sum = 0.0f;
          const float32_t *local_weight_ptr =
              weight_ptr + (f - ch_out_start) * kernel_size;

          for (uint32_t k = 0; k < kernel_size; k++) {
            sum += im2col_buffer[k] * local_weight_ptr[k];
          }

          uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;

          pDstC[out_idx] = sum;
        }
      }
    }
  }
}

void PULP_ConvTrans2d_fp32_fp32_fp32_HWC(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t F_total, const float *__restrict__ pWeight, uint32_t C, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ, float *__restrict__ pGradIn,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C);
  uint16_t ch_count = ch_stop - ch_start;

  if (ch_count == 0) {
    return;
  }

  uint32_t H_in = (H_out - 1) * SP + P - pad_top - pad_bottom;
  uint32_t W_in = (W_out - 1) * SQ + Q - pad_left - pad_right;

  for (uint32_t ih = 0; ih < H_in; ++ih) {
    for (uint32_t iw = 0; iw < W_in; ++iw) {
      uint32_t gi_base = (ih * W_in + iw) * C;
      for (uint32_t ic = ch_start; ic < ch_stop; ++ic) {
        pGradIn[gi_base + ic] = 0.0f;
      }
    }
  }

  for (uint32_t ic = ch_start; ic < ch_stop; ++ic) {

    uint32_t oc = ic;

    for (uint32_t kh = 0; kh < P; ++kh) {
      for (uint32_t kw = 0; kw < Q; ++kw) {

        uint32_t w_idx = ic * (P * Q) + kh * Q + kw;

        for (uint32_t oh = 0; oh < H_out; ++oh) {
          int32_t ih =
              (int32_t)oh * (int32_t)SP + (int32_t)kh - (int32_t)pad_top;

          if (ih < 0 || ih >= (int32_t)H_in)
            continue;

          for (uint32_t ow = 0; ow < W_out; ++ow) {
            int32_t iw =
                (int32_t)ow * (int32_t)SQ + (int32_t)kw - (int32_t)pad_left;

            if (iw < 0 || iw >= (int32_t)W_in)
              continue;

            uint32_t go_idx = (oh * W_out + ow) * C + oc;

            uint32_t gi_idx = ((uint32_t)ih * W_in + (uint32_t)iw) * C + ic;

            pGradIn[gi_idx] += pGradOut[go_idx] * pWeight[w_idx];
          }
        }
      }
    }
  }
}

void PULP_DWConvTrans2d_fp32_fp32_fp32_HWC(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_total, const float *__restrict__ pWeight,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradIn, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right) {

  uint32_t C = C_total;

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C);
  uint16_t ch_count = ch_stop - ch_start;

  if (ch_count == 0) {
    return;
  }

  uint32_t H_in = (H_out - 1) * SP + P - pad_top - pad_bottom;
  uint32_t W_in = (W_out - 1) * SQ + Q - pad_left - pad_right;

  for (uint32_t ih = 0; ih < H_in; ++ih) {
    for (uint32_t iw = 0; iw < W_in; ++iw) {
      uint32_t gi_base = (ih * W_in + iw) * C;
      for (uint32_t ic = ch_start; ic < ch_stop; ++ic) {
        pGradIn[gi_base + ic] = 0.0f;
      }
    }
  }
  for (uint32_t ic = ch_start; ic < ch_stop; ++ic) {

    uint32_t oc = ic; 

    for (uint32_t kh = 0; kh < P; ++kh) {
      for (uint32_t kw = 0; kw < Q; ++kw) {

        uint32_t w_idx = ic * (P * Q) + kh * Q + kw;
        float w_val = pWeight[w_idx];

        for (uint32_t oh = 0; oh < H_out; ++oh) {
          int32_t ih =  (int32_t)oh * (int32_t)SP + (int32_t)kh - (int32_t)pad_top;
          if (ih < 0 || ih >= (int32_t)H_in) continue;

          for (uint32_t ow = 0; ow < W_out; ++ow) {
            int32_t iw =  (int32_t)ow * (int32_t)SQ + (int32_t)kw - (int32_t)pad_left;
            if (iw < 0 || iw >= (int32_t)W_in) continue;

            uint32_t go_idx = (oh * W_out + ow) * C + oc;
            uint32_t gi_idx = ((uint32_t)ih * W_in + (uint32_t)iw) * C + ic;
            
            // Workaround for GCC/RISC-V compiler optimization bug
            // Without this printf, the compiler generates incorrect pointer arithmetic
            // causing wrong results at specific indices (w=0,1 positions)
            
            pGradIn[gi_idx] += pGradOut[go_idx] * w_val;
          }
        }
      }
    }
  }
}

void PULP_ConvGradW2d_fp32_fp32_fp32_NCHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C_out >> log2Core) + ((C_out & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C_out);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C_out);

  if (ch_start >= ch_stop) {
    return;
  }

  // Compute weight gradients
  for (uint32_t oc = ch_start; oc < ch_stop; ++oc) {
    for (uint32_t ic = 0; ic < C_in; ++ic) {
      for (uint32_t kh = 0; kh < P; ++kh) {
        for (uint32_t kw = 0; kw < Q; ++kw) {

          float grad_sum = 0.0f;
          int valid_count = 0;

          for (uint32_t oh = 0; oh < H_out; ++oh) {
            for (uint32_t ow = 0; ow < W_out; ++ow) {

              int32_t ih =
                  (int32_t)oh * (int32_t)SP + (int32_t)kh - (int32_t)pad_top;
              int32_t iw =
                  (int32_t)ow * (int32_t)SQ + (int32_t)kw - (int32_t)pad_left;

              if (ih >= 0 && ih < (int32_t)H_in && iw >= 0 &&
                  iw < (int32_t)W_in) {
                uint32_t go_idx = (oc * H_out + oh) * W_out + ow;
                float gy = pGradOut[go_idx];

                uint32_t in_idx = (ic * H_in + ih) * W_in + iw;
                float x = pInput[in_idx];

                grad_sum += gy * x;
                valid_count++;
              }
            }
          }

          uint32_t gw_idx = ((oc * C_in + ic) * P + kh) * Q + kw;
          pGradWeight[gw_idx] = grad_sum;
        }
      }
    }
  }
}

void PULP_ConvGradB2d_fp32_fp32_NCHW(const float *__restrict__ pGradOut,
                                     uint32_t H_out, uint32_t W_out,
                                     uint32_t C_out,
                                     float *__restrict__ pGradBias) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C_out >> log2Core) + ((C_out & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C_out);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C_out);

  if (ch_start >= ch_stop) {
    return;
  }

  // Compute bias gradients
  // For each output channel, sum all gradients across batch, height, and width
  for (uint32_t oc = ch_start; oc < ch_stop; ++oc) {
    float grad_sum = 0.0f;

    // Sum over all spatial positions
    for (uint32_t oh = 0; oh < H_out; ++oh) {
      for (uint32_t ow = 0; ow < W_out; ++ow) {
        // NCHW layout: [oc, oh, ow]
        uint32_t go_idx = (oc * H_out + oh) * W_out + ow;
        grad_sum += pGradOut[go_idx];
      }
    }

    pGradBias[oc] = grad_sum;
  }
}