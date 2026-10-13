#ifndef __DEEPLOY_BASIC_MATH_SCATTERND_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SCATTERND_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void ScatterND_fp32(const float32_t *__restrict__ data,
                    const int64_t *__restrict__ indices,
                    int32_t indices_offset_deeploy,
                    const float32_t *__restrict__ updates,
                    float32_t *__restrict__ output,
                    const int64_t *__restrict__ d, int32_t n_idx, int32_t n_kr,
                    int32_t k, int32_t data_size);

void ScatterND_s8(const int8_t *__restrict__ data,
                  const int64_t *__restrict__ indices,
                  int32_t indices_offset_deeploy,
                  const int8_t *__restrict__ updates,
                  int8_t *__restrict__ output, const int64_t *__restrict__ d,
                  int32_t n_idx, int32_t n_kr, int32_t k, int32_t data_size);

#endif //__DEEPLOY_BASIC_MATH_SCATTERND_KERNEL_HEADER_
