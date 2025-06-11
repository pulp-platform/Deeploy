#include "DeeployBasicMath.h"
#include <math.h>

void MaxPool1d_fp32_fp32(float32_t const *__restrict__ pSrcA, uint32_t C,
                         uint32_t W, uint32_t K, uint32_t S,
                         float32_t *__restrict__ pDstC) {
  uint32_t W_out = (W - K) / S + 1;
  for (uint32_t c = 0; c < C; ++c) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      float32_t max = -INFINITY;
      for (uint32_t k = 0; k < K; ++k) {
        uint32_t w_in = w_out * S + k;
        if (w_in >= W)
          continue;
        float32_t tmp = pSrcA[c * W + w_in];
        if (tmp > max) {
          max = tmp;
        }
      }
      pDstC[c * W_out + w_out] = max;
    }
  }
}
