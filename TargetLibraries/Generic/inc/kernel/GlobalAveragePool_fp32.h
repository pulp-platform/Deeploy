/* =====================================================================
 * Title:        GlobalAveragePool.h
 * Description:  Header for GlobalAveragePool kernels
 *
 * Date:         18.08.2025
 *
 * ===================================================================== */

#ifndef __DEEPLOY_BASIC_MATH_GLOBAL_AVERAGE_POOL_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_GLOBAL_AVERAGE_POOL_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void GlobalAveragePool_fp32_NCHW(const float32_t *__restrict__ pSrc, uint32_t N,
                                uint32_t C, uint32_t H, uint32_t W,
                                float32_t *__restrict__ pDst);

#endif //__DEEPLOY_BASIC_MATH_GLOBAL_AVERAGE_POOL_KERNEL_HEADER_