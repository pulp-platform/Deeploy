#ifndef __DEEPLOY_BASIC_MATH_GRIDSAMPLE_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_GRIDSAMPLE_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void GridSample_fp32_fp32(const float32_t *__restrict__ pSrcA,
                          const float32_t *__restrict__ pSrcB,
                          float32_t *__restrict__ pDstC, int32_t *inputA,
                          int32_t shapeA_len, int32_t *inputB,
                          int32_t shapeB_len);

#endif //__DEEPLOY_BASIC_MATH_GRIDSAMPLE_KERNEL_HEADER_
