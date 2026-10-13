#include "DeeployBasicMath.h"
#include <math.h>

void Cos_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
  for (int i = 0; i < dataSize; i++) {
    data_out[i] = cosf(data_in[i]);
  }
}
