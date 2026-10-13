#include "DeeployBasicMath.h"
#include <math.h>

void Erf_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
  for (int i = 0; i < dataSize; i++) {
    data_out[i] = erff(data_in[i]);
  }
}
