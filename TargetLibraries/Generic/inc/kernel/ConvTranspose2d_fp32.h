// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#ifndef CONV_TRANSPOSE2D_FP32_H
#define CONV_TRANSPOSE2D_FP32_H

#include <stdbool.h>
#include <stdint.h>

// CHW input/output, weights [C_in, C_out / groups, K_h, K_w].
// Output dimensions account for bottom/right padding and output_padding.
void ConvTranspose2d_fp32(const float32_t *input, uint32_t C_in, uint32_t H_in,
                          uint32_t W_in, const float32_t *weight,
                          uint32_t C_out, uint32_t groups, uint32_t K_h,
                          uint32_t K_w, uint32_t stride_h, uint32_t stride_w,
                          uint32_t dilation_h, uint32_t dilation_w,
                          uint32_t pad_top, uint32_t pad_left,
                          const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t H_out, uint32_t W_out);

#endif // CONV_TRANSPOSE2D_FP32_H
