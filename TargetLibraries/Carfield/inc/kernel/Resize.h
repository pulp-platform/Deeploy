#ifndef CARFIELD_KERNEL_RESIZE_H
#define CARFIELD_KERNEL_RESIZE_H

#include <stdint.h>
#include "Types.h"

void Carfield_ResizeNearest2D(
    float32_t *input,
    float32_t *output,
    int32_t    C,
    int32_t    H_in,
    int32_t    W_in,
    int32_t    H_out,
    int32_t    W_out,
    int32_t    scale_h,
    int32_t    scale_w,
    int32_t core_id,
    int32_t numThreads
);

#endif
