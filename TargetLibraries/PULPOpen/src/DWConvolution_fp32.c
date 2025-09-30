/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer) {

  // Compute core information
  int8_t core_id = pi_core_id();
  int8_t log2Core = (int8_t)log2(NUM_CORES);

  // Compute the chunk size for each core
  // (Splitting work along the output channels)
  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  // If there is no output channel to process, return
  // (when F < NUM_CORES and working on a core with id > F)
  if (ch_out_count == 0) {
    return;
  }

  // Move pointer of the weights for the current core
  const float32_t *weight_ptr = pSrcB + ch_out_start * P * Q;

  // Move pointer of the im2col buffer for the current core
  uint32_t im2col_size_per_core = P * Q;
  float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

  // Compute the output dimensions
  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  uint32_t kernel_size = P * Q * F_total;

  // Compute the output
  if (has_bias) {
    // Work on individual output elements
    // (each element depends on a column from the im2col buffer
    // and one convolutional filter, stored in memory continuously)
    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        // Compute height and width starting point
        // (depending on stride and padding)
        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        // Initialize the padded part of the im2col buffer with 0
        // Work on the TOP padding
        for (int32_t h_in = (int32_t)h_in_start;
             h_in < MIN(0, (int32_t)(h_in_start + P)); h_in++) {
          for (int32_t w_in = (int32_t)w_in_start;
               w_in < (int32_t)(w_in_start + Q); w_in++) {
            im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
          }
        }

        // Work on the BOTTOM padding
        for (uint32_t h_in = MAX(H, h_in_start); h_in < h_in_start + P;
             h_in++) {
          for (int32_t w_in = (int32_t)w_in_start;
               w_in < (int32_t)(w_in_start + Q); w_in++) {
            im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
          }
        }

        // Work on the remaining LEFT padding
        for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
             h_in++) {
          for (int32_t w_in = (int32_t)w_in_start;
               w_in < MIN(0, (int32_t)(w_in_start + Q)); w_in++) {
            im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
          }
        }

        // Work on the remaining RIGHT padding
        for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
             h_in++) {
          for (uint32_t w_in = MAX(W, w_in_start); w_in < w_in_start + Q;
               w_in++) {
            im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
          }
        }

        // Copy input data to im2col buffer
        // Input channels depend on the output channels assigned to the core
        // (each input channel is associated with F_total / C output channels,
        // number which corresponds to the "group" parameter in the Conv ONNX
        // operator)
        for (uint32_t c = ch_out_start / (F_total / C);
             c < (ch_out_stop + 1) / (F_total / C); c++) {
          // Copy the valid input data to the im2col buffer
          for (uint32_t h_in = MAX(0, h_in_start);
               h_in < MIN(H, h_in_start + P); h_in++) {
            for (uint32_t w_in = MAX(0, w_in_start);
                 w_in < MIN(W, w_in_start + Q); w_in++) {
              uint32_t in_idx = (h_in * W + w_in) * C + c;
              im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] =
                  pSrcA[in_idx];
            }
          }

          // Compute output channels of interest, based on current input channel
          // and core
          uint32_t lower_f, upper_f;

          if (c * (F_total / C) < ch_out_start) {
            lower_f = ch_out_start;
          } else {
            lower_f = c * (F_total / C);
          }

          if ((c + 1) * (F_total / C) < ch_out_stop) {
            upper_f = (c + 1) * (F_total / C);
          } else {
            upper_f = ch_out_stop;
          }

          // Perform convolution for the assigned output channels
          for (uint32_t f = lower_f; f < upper_f; f++) {
            float32_t sum = 0.0f;
            uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;

            for (uint32_t im2col_idx = 0; im2col_idx < P * Q; im2col_idx++) {
              sum +=
                  im2col_buffer[im2col_idx] *
                  weight_ptr[(f - ch_out_start) * P * Q + im2col_idx % (P * Q)];
            }

            // Copy the result to the output tensor
            pDstC[out_idx] = sum + pSrcBias[f];
          }
        }
      }
    }
  } else {
    // Work on individual output elements
    // (each element depends on a column from the im2col buffer
    // and one convolutional filter, stored in memory continuously)
    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        // Compute height and width starting point
        // (depending on stride and padding)
        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        // Initialize the padded part of the im2col buffer with 0
        // Work on the TOP padding
        for (int32_t h_in = (int32_t)h_in_start;
             h_in < MIN(0, (int32_t)(h_in_start + P)); h_in++) {
          for (int32_t w_in = (int32_t)w_in_start;
               w_in < (int32_t)(w_in_start + Q); w_in++) {
            im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
          }
        }

        // Work on the BOTTOM padding
        for (uint32_t h_in = MAX(H, h_in_start); h_in < h_in_start + P;
             h_in++) {
          for (int32_t w_in = (int32_t)w_in_start;
               w_in < (int32_t)(w_in_start + Q); w_in++) {
            im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
          }
        }

        // Work on the remaining LEFT padding
        for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
             h_in++) {
          for (int32_t w_in = (int32_t)w_in_start;
               w_in < MIN(0, (int32_t)(w_in_start + Q)); w_in++) {
            im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
          }
        }

        // Work on the remaining RIGHT padding
        for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
             h_in++) {
          for (uint32_t w_in = MAX(W, w_in_start); w_in < w_in_start + Q;
               w_in++) {
            im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
          }
        }

        // Copy input data to im2col buffer
        // Input channels depend on the output channels assigned to the core
        // (each input channel is associated with F_total / C output channels,
        // number which corresponds to the "group" parameter in the Conv ONNX
        // operator)
        for (uint32_t c = ch_out_start / (F_total / C);
             c < (ch_out_stop + 1) / (F_total / C); c++) {
          // Copy the valid input data to the im2col buffer
          for (uint32_t h_in = MAX(0, h_in_start);
               h_in < MIN(H, h_in_start + P); h_in++) {
            for (uint32_t w_in = MAX(0, w_in_start);
                 w_in < MIN(W, w_in_start + Q); w_in++) {
              uint32_t in_idx = (h_in * W + w_in) * C + c;
              im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] =
                  pSrcA[in_idx];
            }
          }

          // Compute output channels of interest, based on current input channel
          // and core
          uint32_t lower_f, upper_f;

          if (c * (F_total / C) < ch_out_start) {
            lower_f = ch_out_start;
          } else {
            lower_f = c * (F_total / C);
          }

          if ((c + 1) * (F_total / C) < ch_out_stop) {
            upper_f = (c + 1) * (F_total / C);
          } else {
            upper_f = ch_out_stop;
          }

          // Perform convolution for the assigned output channels
          for (uint32_t f = lower_f; f < upper_f; f++) {
            float32_t sum = 0.0f;
            uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;

            for (uint32_t im2col_idx = 0; im2col_idx < P * Q; im2col_idx++) {
              sum +=
                  im2col_buffer[im2col_idx] *
                  weight_ptr[(f - ch_out_start) * P * Q + im2col_idx % (P * Q)];
            }

            // Copy the result to the output tensor
            pDstC[out_idx] = sum;
          }
        }
      }
    }
  }

  return;
}
