#ifndef __DEEPLOY_BASIC_MATH_SPLIT_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SPLIT_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Split_fp32(const float32_t *__restrict__ pSrcA, int32_t *inputA,
                int32_t shapeA_len, const int64_t *__restrict__ pSrcB,
                int32_t sizeB, int64_t offset, int32_t axis_in,
                float32_t **__restrict__ pDstC, int32_t num_output);

#endif //__DEEPLOY_BASIC_MATH_SPLIT_KERNEL_HEADER_
