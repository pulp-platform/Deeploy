#include "Div.h"

#ifdef __riscv
extern float32_t __divsf3(float32_t, float32_t);
#define CARFIELD_FDIV(a, b) __divsf3((a), (b))
#else
#define CARFIELD_FDIV(a, b) ((a) / (b))
#endif

void Carfield_Div(float32_t *data_in_1, float32_t *data_in_2, float32_t *data_out,
                  uint32_t size, uint32_t channels, uint32_t spatialSize,
                  uint32_t core_id, uint32_t numThreads) {

    uint32_t chunk = (size + numThreads - 1) / numThreads;
    uint32_t start = core_id * chunk;
    uint32_t end   = start + chunk;
    if (end > size) end = size;
    if (start > size) start = size;

    for (uint32_t i = start; i < end; i++) {
        uint32_t c = (i / spatialSize) % channels;
        float32_t x = data_in_1[i];
        float32_t d = data_in_2[c];
        data_out[i] = CARFIELD_FDIV(x, d);
    }
}
