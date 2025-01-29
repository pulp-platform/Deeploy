#include "DeeploySnitchMath.h"

void Softmax_fp32(float32_t *input, float32_t *output, int32_t ldI,
                  int32_t batch_offset, int32_t batch_size, int32_t seq_len,
                  int32_t input_samples) {

  float32_t max_core = 0.0; // max value of the current core
  float32_t sum = 0.0;      // sum of the exp values of the current core
  int32_t compute_id = snrt_global_compute_core_idx();
  int32_t row_offset = compute_id * input_samples;
  for (int32_t b = 0; b < batch_size; b++) {
    for (int32_t s = 0; s < seq_len; s++) {
      max_core = -INFINITY;
      sum = 0.0;
      for (int32_t i = 0; i < input_samples; i++) {
        if (input[row_offset + b * batch_offset + s * ldI + i] > max_core) {
          max_core = input[row_offset + b * batch_offset + s * ldI + i];
        }
      }
      // compute the shifted value of the current row
      for (int32_t i = 0; i < input_samples; i++) {
        output[row_offset + b * batch_offset + s * ldI + i] =
            expf(input[row_offset + b * batch_offset + s * ldI + i] - max_core);
        sum += output[row_offset + b * batch_offset + s * ldI + i];
      }
      // compute the softmax value of the current row
      for (int32_t i = 0; i < input_samples; i++) {
        output[row_offset + b * batch_offset + s * ldI + i] /= sum;
      }
    }
  }
}

// Softmax_fp32_opt currently not supported by current compiler
/*
typedef __fp16 v2f16 __attribute__((vector_size(4)));
void Softmax_fp32_opt(float *input, float *output, int32_t ldI,
                      int32_t batch_offset, int32_t batch_size, int32_t seq_len,
                      int32_t input_samples) {
  float max_core = 0.0; // max value of the current core
  float sum = 0.0;      // sum of the exp values of the current core
  uint32_t num = 0xffff;
  v2f16 offset_lut = {0.0, 0.0};
  v2f16 scale_lut = {0.0, 0.0};
  float tmp = 0.0;

  for (int32_t b = 0; b < batch_size; b++) {
    for (int32_t s = 0; s < seq_len; s++) {
      max_core = -INFINITY;
      sum = 0.0;
      // TO CHECK
      for (int32_t i = 0; i < input_samples; i++) {
        if (input[b * batch_offset + s * ldI + i] > max_core) {
          max_core = input[b * batch_offset + s * ldI + i];
        }
      }
      // compute the shifted value of the current row
      for (int32_t i = 0; i < input_samples; i++) {
        float inp = input[b * batch_offset + s * ldI + i] - max_core;
        asm volatile("fcvt.h.s fa0, %[input] \n"
                     "fmadd.h %[offset], fa0, %[scale], %[offset] \n"
                     "fcvt.s.h %[tmp], %[offset] \n"
                     : [offset] "+f"(offset_lut), [tmp] "+f"(tmp)
                     : [input] "f"(inp), [scale] "f"(scale_lut)
                     : "ft0", "ft1", "ft2", "fa0");
        output[b * batch_offset + s * ldI + i] = tmp;
        sum += output[b * batch_offset + s * ldI + i];
      }
      // compute the softmax value of the current row
      for (int32_t i = 0; i < input_samples; i++) {
        output[b * batch_offset + s * ldI + i] /= sum;
        //  printf("Outputs %f \n",output[b * batch_offset + s * ldI + i]);
      }
    }
  }
}
*/