#include "Sub.h"

void Carfield_Sub(float32_t *data_in_1, float32_t *data_in_2, float32_t *data_out,
                  uint32_t size, uint32_t channels, uint32_t spatialSize,
                  uint32_t core_id, uint32_t numThreads) {

    uint32_t chunk = (size + numThreads - 1) / numThreads;
    uint32_t start = core_id * chunk;
    uint32_t end   = start + chunk;
    if (end > size) end = size;
    if (start > size) start = size;

    for (uint32_t i = start; i < end; i++) {
        uint32_t c = (i / spatialSize) % channels;
        data_out[i] = data_in_1[i] - data_in_2[c];
    }
}
