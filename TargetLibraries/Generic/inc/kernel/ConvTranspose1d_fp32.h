#ifndef CONV_TRANSPOSE1D_FP32_H
#define CONV_TRANSPOSE1D_FP32_H

#include <stdbool.h>
#include <stdint.h>

void ConvTranspose1d_fp32(const float32_t *input, uint32_t C_in, uint32_t W_in,
                          const float32_t *weight, uint32_t C_out, uint32_t K,
                          uint32_t stride, const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t W_out);

#endif // CONV_TRANSPOSE1D_FP32_H
