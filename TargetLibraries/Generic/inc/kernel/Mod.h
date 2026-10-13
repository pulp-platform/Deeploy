#ifndef __DEEPLOY_BASIC_MATH_MOD_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_MOD_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Mod_fp32_fp32(const float32_t *__restrict__ pSrcA,
                   const float32_t *__restrict__ pSrcB,
                   int8_t fmod, // NOT USED
                   float32_t *__restrict__ pDstC, int32_t *inputA,
                   int32_t shapeA_len, int32_t *inputB, int32_t shapeB_len);

void Mod_s8_s8(const int8_t *__restrict__ pSrcA, int8_t A_offset,
               const int8_t *__restrict__ pSrcB, int8_t B_offset,
               int8_t fmod, // NOT USED
               int8_t *__restrict__ pDstC, int8_t C_offset, int32_t *inputA,
               int32_t shapeA_len, int32_t *inputB, int32_t shapeB_len);

void Mod_s64_s64(const int64_t *__restrict__ pSrcA, int64_t A_offset,
                 const int64_t *__restrict__ pSrcB, int64_t B_offset,
                 int8_t fmod, // NOT USED
                 int64_t *__restrict__ pDstC, int64_t C_offset, int32_t *inputA,
                 int32_t shapeA_len, int32_t *inputB, int32_t shapeB_len);

#endif //__DEEPLOY_BASIC_MATH_MOD_KERNEL_HEADER_
