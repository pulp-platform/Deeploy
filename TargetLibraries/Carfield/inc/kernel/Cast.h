#ifndef CARFIELD_KERNEL_CAST_H
#define CARFIELD_KERNEL_CAST_H

#include <stdint.h>
#include "Types.h"

void Carfield_CastInt64ToInt32(
    int64_t *input,
    int32_t *output,
    int32_t  size,
    int32_t core_id,
    int32_t numThreads
);

#endif
