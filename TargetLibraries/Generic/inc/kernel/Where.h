#ifndef __DEEPLOY_BASIC_MATH_WHERE_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_WHERE_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Where_fp32_fp32(const int8_t *__restrict__ pSrcCond,
                     int32_t offset, // don't know how to create BOOL_T datatype
                     const float32_t *__restrict__ pSrcX,
                     const float32_t *__restrict__ pSrcY, int32_t *inputCond,
                     int32_t shapeCond_len, int32_t *inputX, int32_t shapeX_len,
                     int32_t *inputY, int32_t shapeY_len,
                     float32_t *__restrict__ pDstC);

#endif //__DEEPLOY_BASIC_MATH_WHERE_KERNEL_HEADER_
