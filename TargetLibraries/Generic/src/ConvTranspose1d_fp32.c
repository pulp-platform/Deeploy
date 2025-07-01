
#include "DeeployBasicMath.h"

void ConvTranspose1d_fp32(const float32_t *input, uint32_t C_in, uint32_t W_in,
                          const float32_t *weight, uint32_t C_out, uint32_t K,
                          uint32_t stride, const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t W_out) {
  /*
  input:       [C_in, W_in]
  weight:      [C_in, C_out, K]  
  output:      [C_out, W_out] 
  bias:        [C_out] optionally
  
  */

  // Output initialization
  for (uint32_t c = 0; c < C_out; ++c) {
    for (uint32_t w = 0; w < W_out; ++w) {
      output[c * W_out + w] = 0.0f;
    }
  }

  // For each output channel
  for (uint32_t cout = 0; cout < C_out; ++cout) {
    // For each input channel
    for (uint32_t cin = 0; cin < C_in; ++cin) {
      // For each input width
      for (uint32_t w_in = 0; w_in < W_in; ++w_in) {
        float32_t val = input[cin * W_in + w_in];
        // Transposed convolution: output width is calculated based on stride
        for (uint32_t k = 0; k < K; ++k) {
          uint32_t w_out = w_in * stride + k;
          if (w_out < W_out) {
            // weight indexing: weight[cin, cout, k]
            float32_t wgt = weight[cin * (C_out * K) + cout * K + k];
            output[cout * W_out + w_out] += val * wgt;
          }
        }
      }
    }
    if (has_bias) {
      for (uint32_t w = 0; w < W_out; ++w) {
        output[cout * W_out + w] += bias[cout];
      }
    }
  }
}