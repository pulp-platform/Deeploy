#ifndef __DEEPLOY_BASIC_MATH_GATHERELEMENTS_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_GATHERELEMENTS_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void GatherElements_fp32_s64(const float32_t *__restrict__ pSrcA,
                             int32_t *inputA, int32_t shapeA_len,
                             const int64_t *__restrict__ pSrcB, int32_t *inputB,
                             int32_t shapeB_len, int64_t offset, int32_t axis,
                             float32_t *__restrict__ pDstC);

void GatherElements_fp32_s32(const float32_t *__restrict__ pSrcA,
                             int32_t *inputA, int32_t shapeA_len,
                             const int32_t *__restrict__ pSrcB, int32_t *inputB,
                             int32_t shapeB_len, int32_t offset, int32_t axis,
                             float32_t *__restrict__ pDstC);

#endif //__DEEPLOY_BASIC_MATH_GATHERELEMENTS_KERNEL_HEADER_
