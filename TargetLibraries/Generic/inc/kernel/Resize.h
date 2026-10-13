#ifndef __DEEPLOY_BASIC_MATH_RESIZE_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_RESIZE_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Resize_fp32_fp32(float32_t *data_in, int32_t *inputA, int32_t shapeA_len,
                      float32_t *scale, int32_t scale_len, float32_t *data_out,
                      int8_t coord, int8_t mode, int8_t nearest_mode);

#endif //__DEEPLOY_BASIC_MATH_RESIZE_KERNEL_HEADER_
