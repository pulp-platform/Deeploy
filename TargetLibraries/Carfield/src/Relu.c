
#include "Relu.h"
#include "Utils.h"

void Carfield_Relu(float32_t *input, float32_t *output, uint32_t size, uint32_t core_id, uint32_t numThreads) {

    uint32_t chunk = (size + numThreads - 1) / numThreads;
    uint32_t start  = core_id * chunk;
    uint32_t end    = start + chunk;
    if (end > size) end = size;          // Last core can have less work to do
    if (start > size) start = size;      // With size < numThreads, some cores will have no work to do

    for (uint32_t i = start; i < end; i++) {
        float32_t x = input[i];          // to avoid double loading from DDR because of the macro
        output[i] = __MAX(x, 0.0f);
    }
}
