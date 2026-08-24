// SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#ifndef CONV_TRANSPOSE_FP32_H
#define CONV_TRANSPOSE_FP32_H

#include <stdbool.h>
#include <stdint.h>

void ConvTranspose1d_fp32(const float32_t *input, uint32_t C_in, uint32_t W_in,
                          const float32_t *weight, uint32_t C_out, uint32_t K,
                          uint32_t stride, const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t W_out);

void ConvTranspose2d_fp32(const float32_t *input, uint32_t C_in, uint32_t H_in,
                          uint32_t W_in, const float32_t *weight,
                          uint32_t C_out, uint32_t kH, uint32_t kW,
                          uint32_t stride_h, uint32_t stride_w,
                          const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t H_out, uint32_t W_out);

#endif // CONV_TRANSPOSE_FP32_H
