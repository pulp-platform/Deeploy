#include "DeeployBasicMath.h"



void BatchNorm_fp32(
    const float32_t *input,
    const float32_t *gamma,
    const float32_t *beta,
    const float32_t *mean,
    const float32_t *var,
    float32_t *output,
    int N
) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < N; i++) {
        float32_t norm = (input[i] - mean[i]) / sqrtf(var[i] + epsilon);
        output[i] = gamma[i] * norm + beta[i];
    }
}
