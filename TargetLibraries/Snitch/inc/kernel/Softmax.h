#include "DeeploySnitchMath.h"

void softmax_fp32(float *input, float *output, int32_t ldI,
                  int32_t batch_offset, int32_t batch_size, int32_t seq_len,
                  int32_t input_samples);