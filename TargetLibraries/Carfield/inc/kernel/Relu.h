#ifndef CARFIELD_KERNEL_RELU_H
#define CARFIELD_KERNEL_RELU_H

#include <stdint.h>
#include "Types.h"

void Carfield_Relu(
    float32_t *input,
    float32_t *output,
    uint32_t    size,
    uint32_t core_id,
    uint32_t numThreads
);

#endif // CARFIELD_KERNEL_RELU_H
