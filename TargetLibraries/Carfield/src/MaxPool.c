#include "MaxPool.h"

void Carfield_MaxPool2D(float32_t *input, float32_t *output,
                        int32_t C, int32_t H_in, int32_t W_in,
                        int32_t H_out, int32_t W_out,
                        int32_t P, int32_t Q,
                        int32_t stride_h, int32_t stride_w,
                        int32_t core_id, int32_t numThreads) {

    int32_t plane = H_out * W_out;
    int32_t size  = C * plane;

    int32_t chunk = (size + numThreads - 1) / numThreads;
    int32_t start = core_id * chunk;
    int32_t end   = start + chunk;
    if (end > size) end = size;
    if (start > size) start = size;

    for (int32_t o = start; o < end; o++) {
        int32_t c = o / plane;
        int32_t r = o - c * plane;
        int32_t h = r / W_out;
        int32_t w = r - h * W_out;

        int32_t base = c * H_in * W_in + (h * stride_h) * W_in + (w * stride_w);

        float32_t acc = input[base];

        for (int32_t q = 1; q < Q; q++) {
            float32_t v = input[base + q];
            if (v > acc) acc = v;
        }
        for (int32_t p = 1; p < P; p++) {
            for (int32_t q = 0; q < Q; q++) {
                float32_t v = input[base + p * W_in + q];
                if (v > acc) acc = v;
            }
        }

        output[o] = acc;
    }
}
