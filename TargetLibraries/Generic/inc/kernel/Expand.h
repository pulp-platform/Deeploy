#ifndef __DEEPLOY_BASIC_MATH_EXPAND_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_EXPAND_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Expand_fp32(const float32_t *__restrict__ pSrcA,
                 float32_t *__restrict__ pDstC,
                 const int64_t *__restrict__ pSrcB, int32_t shapeB_len,
                 int64_t offset, int32_t *inputA, int32_t shapeA_len);

void Expand_s8(const int8_t *__restrict__ pSrcA, int8_t A_offset,
               int8_t *__restrict__ pDstC, int8_t C_offset,
               const int8_t *__restrict__ pSrcB, int32_t shapeB_len,
               int8_t offset, int32_t *inputA, int32_t shapeA_len);

void Expand_s64(const int64_t *__restrict__ pSrcA, int64_t A_offset,
                int64_t *__restrict__ pDstC, int64_t C_offset,
                const int64_t *__restrict__ pSrcB, int32_t shapeB_len,
                int64_t offset, int32_t *inputA, int32_t shapeA_len);

#endif //__DEEPLOY_BASIC_MATH_EXPAND_KERNEL_HEADER_
