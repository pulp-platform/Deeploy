#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <stdbool.h>
#include <stdint.h>

void BatchNorm_fp32(const float32_t *input, const float32_t *gamma,
                    const float32_t *beta, const float32_t *mean,
                    const float32_t *var, float32_t *output, int N, int C,
                    int L);

#endif // BATCHNORM_H
