#ifndef __DEEPLOY_BASIC_MATH_COS_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_COS_KERNEL_HEADER_

#include "DeeployBasicMath.h"

// Cosinus calculated as radians between -1 and 1
void Cos_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize);

#endif //__DEEPLOY_BASIC_MATH_COS_KERNEL_HEADER_
