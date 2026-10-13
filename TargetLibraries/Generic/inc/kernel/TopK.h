#ifndef __DEEPLOY_BASIC_MATH_TOPK_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_TOPK_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void TopK_fp32(const float32_t *__restrict__ pSrcA,
               float32_t *__restrict__ pDstC, int64_t *__restrict__ pDstI,
               int64_t I_offset, // indices, maybe no offset
               const int64_t *__restrict__ K, int64_t K_offset, int32_t in_axis,
               int32_t *inputA, int32_t shapeA_len);

#endif //__DEEPLOY_BASIC_MATH_TOPK_KERNEL_HEADER_
