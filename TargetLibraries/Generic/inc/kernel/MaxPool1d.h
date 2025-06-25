#ifndef __DEEPLOY_BASIC_MATH_MAXPOOL1D_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_MAXPOOL1D_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void MaxPool1d_fp32_fp32(float32_t const *__restrict__ pSrcA, uint32_t C,
                         uint32_t W, uint32_t K, uint32_t S,
                         float32_t *__restrict__ pDstC);

#endif