#ifndef __DEEPLOY_BASIC_MATH_TILE_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_TILE_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Tile_fp32(const float32_t *__restrict__ pSrcA,
               float32_t *__restrict__ pDstC, const int64_t *__restrict__ pSrcB,
               int32_t shapeB_len, int64_t offset, int32_t *inputA,
               int32_t shapeA_len);

void Tile_s64(const int64_t *__restrict__ pSrcA, int64_t *__restrict__ pDstC,
              const int64_t *__restrict__ pSrcB, int32_t shapeB_len,
              int64_t offset, int32_t *inputA, int32_t shapeA_len);
#endif //__DEEPLOY_BASIC_MATH_TILE_KERNEL_HEADER_
