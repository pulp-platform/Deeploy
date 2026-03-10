/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"
#include <math.h>
#include "perf_utils.h"

// TODO: 1) loop unrolling for ILP perf
// TODO: 2) Perturbation directly integrated in GEMM or Conv kernels.
/* --------------------------- RNG ---------------------------------- */

uint32_t Xorshift32(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

/* --------------------------- Samplers ---------------------------------- */

float32_t TriangularSample(uint32_t *state) {
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    // mutate state to avoid same seed for u2.
    *state = Xorshift32(*state);
    float32_t u2 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return u1 - u2;
}

float32_t UniformSample(uint32_t *state) {
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return u1-0.5f; // centered around 0
}

float32_t GaussianSample(uint32_t *state) {
    // Box-Muller transform
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in (0,1]
    // mutate state to avoid same seed for u2.
    *state = Xorshift32(*state);
    float32_t u2 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float32_t)PI_F * u2);
}

/* ---------------- Ziggurat method for Gaussian sampling ---------------- */
// This implementation is adapted from the public domain Ziggurat algorithm
// by Marsaglia and Tsang.

void build_ziggurat_tables() {
    if (ziggurat_tables_initialized) return;

    float32_t dn = (float32_t) ZIGGURAT_R;
    float32_t tn = dn;
    float32_t vn = (float32_t) ZIGGURAT_V;

    // Set up the tables
    float32_t q = vn / expf(-0.5f * dn * dn);
    kn[0] = (uint32_t)((dn / q) * (float32_t)0xFFFFFFFF);
    kn[1] = 0;

    wn[0] = (float32_t)(q / (float32_t)0xFFFFFFFF);
    wn[ZIGGURAT_TABLE_SIZE - 1] = (float32_t)(dn / (float32_t)0xFFFFFFFF);

    fn[0] = 1.0f;
    fn[ZIGGURAT_TABLE_SIZE - 1] = expf(-0.5f * dn * dn);

    for (uint32_t i = ZIGGURAT_TABLE_SIZE - 2; i >= 1; i--) {
        dn = sqrtf(-2.0f * logf(vn / dn + expf(-0.5f * dn * dn)));
        kn[i + 1] = (uint32_t)((dn / tn) * (float32_t)0xFFFFFFFF);
        tn = dn;
        fn[i] = expf(-0.5f * dn * dn);
        wn[i] = (float32_t)(dn / (float32_t)0xFFFFFFFF);
    }
    ziggurat_tables_initialized = 1;
}


float32_t GaussianZigguratSample(uint32_t *state) {
    if (!ziggurat_tables_initialized) {
        build_ziggurat_tables();
    }

    int32_t hz;
    uint32_t iz;
    float32_t x, y;

    for (;;) {
        *state = Xorshift32(*state);
        hz = (int32_t)(*state);
        iz = hz & (ZIGGURAT_TABLE_SIZE - 1);

        // Quick acceptance path
        if ((uint32_t)((hz < 0) ? -hz : hz) < kn[iz])  {
            return (float32_t)hz * wn[iz];
        }

        // Handle the tail
        if (iz == 0) {
            do {
                *state = Xorshift32(*state);
                x = -logf((float32_t)(*state) / (float32_t)0xFFFFFFFF) / ZIGGURAT_R;
                *state = Xorshift32(*state);
                y = -logf((float32_t)(*state) / (float32_t)0xFFFFFFFF);
            } while (y + y < x * x);
            return (hz > 0) ? ZIGGURAT_R + x : -ZIGGURAT_R - x;
        }

        // Slower rejection path
        x = (float32_t)hz * wn[iz];
        if (fn[iz] + ((float32_t)(*state) / (float32_t)0xFFFFFFFF) * (fn[iz - 1] - fn[iz]) < expf(-0.5f * x * x)) {
            return x;
        }
    }
}

void RademacherRNG_init(RademacherRNG *rng, uint32_t seed) {
    rng->state = seed;
    rng->bits = 0;
    rng->bitpos = 32; // force refill on first use
}

float32_t RademacherSample(RademacherRNG *rng) {
    if (rng->bitpos >= 32) {
        rng->state = Xorshift32(rng->state);
        rng->bits = rng->state;
        rng->bitpos = 0;
    }
    float32_t val = (rng->bits & 1) ? 1.0f : -1.0f;
    rng->bits >>= 1;
    rng->bitpos++;
    return val;
}

/* ------------------------- Perturbation Functions -------------------------------- */

void ApplyTrianglePerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon) 
{

    int8_t core_id = pi_core_id();
    int8_t log2Core = LOG2(NUM_CORES);

    perf_stats_t perf_start, perf_end, perf_total;

    // Initialize and start performance counters (only core 0)
    if (core_id == 0) {
        perf_bench_init();
        perf_bench_start();
        perf_bench_read(&perf_start);
    }

    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    if (dir == 0) {epsilon *= -1.0f;}
    for (uint32_t i = 0; i < size; i+=5) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // mutate state to avoid same seed for u2.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i] = pweights[i] + (u1-u2) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // mutate state to avoid same seed for u2.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+1] = pweights[i+1] + (u1-u2) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // mutate state to avoid same seed for u2.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+2] = pweights[i+2] + (u1-u2) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // mutate state to avoid same seed for u2.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+3] = pweights[i+3] + (u1-u2) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // mutate state to avoid same seed for u2.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+4] = pweights[i+4] + (u1-u2) * epsilon;
    }

    if (core_id == 0) {
        perf_bench_stop();
        perf_bench_read(&perf_end);
        perf_bench_diff(&perf_total, &perf_end, &perf_start);

        char label[100];
        snprintf(label, sizeof(label), "Perturb Triangle seed=%u N=%u",
                seed, size);
        perf_bench_print(label, &perf_total);
    }
}

void ApplyUniformPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon)
{

    int8_t core_id = pi_core_id();
    int8_t log2Core = LOG2(NUM_CORES);

    perf_stats_t perf_start, perf_end, perf_total;

    // Initialize and start performance counters (only core 0)
    if (core_id == 0) {
        perf_bench_init();
        perf_bench_start();
        perf_bench_read(&perf_start);
    }

    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    // sqrt(3)*2 factor already included in epsilon to match Gaussian(0, 1) l2 norm.
    if (dir == 0) {epsilon *= -1.0f;}
    for (uint32_t i = 0; i < size; i+=7) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i] = pweights[i] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+1] = pweights[i+1] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+2] = pweights[i+2] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+3] = pweights[i+3] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+4] = pweights[i+4] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+5] = pweights[i+5] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        pweights_dest[i+6] = pweights[i+6] + (u1-0.5f) * epsilon;

        // rng_state ^= rng_state << 13;
        // rng_state ^= rng_state >> 17;
        // rng_state ^= rng_state << 5;
        // u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // pweights_dest[i+7] = pweights[i+7] + (u1-0.5f) * epsilon;

        // rng_state ^= rng_state << 13;
        // rng_state ^= rng_state >> 17;
        // rng_state ^= rng_state << 5;
        // u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // pweights_dest[i+8] = pweights[i+8] + (u1-0.5f) * epsilon;
    }

    if (core_id == 0) {
        perf_bench_stop();
        perf_bench_read(&perf_end);
        perf_bench_diff(&perf_total, &perf_end, &perf_start);

        char label[100];
        snprintf(label, sizeof(label), "Perturb Uniform seed=%u N=%u",
                seed, size);
        perf_bench_print(label, &perf_total);
    }
}

void ApplyGaussianPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    if (dir == 0) {epsilon *= -1.0f;}
    for (uint32_t i = 0; i < size; i++) {
        float32_t u = GaussianZigguratSample(&rng_state);
        pweights_dest[i] = pweights[i] + u * epsilon;
    }
}

void ApplyRademacherPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon) {

    int8_t core_id = pi_core_id();
    int8_t log2Core = LOG2(NUM_CORES);

    perf_stats_t perf_start, perf_end, perf_total;

    // Initialize and start performance counters (only core 0)
    if (core_id == 0) {
        perf_bench_init();
        perf_bench_start();
        perf_bench_read(&perf_start);
    }

    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    if (dir == 0) epsilon *= -1.0f;

    uint32_t n_full_batches = size / 32;
    uint32_t leftover = size % 32;
    uint32_t i = 0;

    // Process full batches
    for (uint32_t batch = 0; batch < n_full_batches; batch++) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < 32; b+=6, i+=6) {
            float32_t r = (bits & 1) ? 1.0f : -1.0f;
            pweights_dest[i]   = pweights[i] + r * epsilon;
            bits >>= 1;
            r = (bits & 1) ? 1.0f : -1.0f;
            pweights_dest[i+1] = pweights[i+1] + r * epsilon;
            bits >>= 1;
        }
    }

    // Process leftover elements
    if (leftover > 0) {
        rng_state = Xorshift32(rng_state);
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < leftover; b++, i++) {
            float32_t r = (bits & 1) ? 1.0f : -1.0f;
            pweights_dest[i] = pweights[i] + r * epsilon;
            bits >>= 1;
        }
    }

    if (core_id == 0) {
        perf_bench_stop();
        perf_bench_read(&perf_end);
        perf_bench_diff(&perf_total, &perf_end, &perf_start);

        char label[100];
        snprintf(label, sizeof(label), "Perturb Rademacher seed=%u N=%u",
                seed, size);
        perf_bench_print(label, &perf_total);
    }
}

void GenEggrollPerturbation(float32_t *__restrict__ p_dest,
                            uint32_t seed,
                            uint32_t size)
{
    // For compatibility with existing codegen templates. Currently maps to Rademacher noise.
    int8_t core_id = pi_core_id();
    int8_t log2Core = LOG2(NUM_CORES);

    perf_stats_t perf_start, perf_end, perf_total;

    // Initialize and start performance counters (only core 0)
    if (core_id == 0) {
        perf_bench_init();
        perf_bench_start();
        perf_bench_read(&perf_start);
    }
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    for (uint32_t i = 0; i < size; i+=5) {
        
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        p_dest[i] = u1-0.5f;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        p_dest[i+1] = u1-0.5f;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        p_dest[i+2] = u1-0.5f;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        p_dest[i+3] = u1-0.5f;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        p_dest[i+4] = u1-0.5f;

        // rng_state ^= rng_state << 13;
        // rng_state ^= rng_state >> 17;
        // rng_state ^= rng_state << 5;
        // u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // p_dest[i+5] = u1-0.5f;

        // rng_state ^= rng_state << 13;
        // rng_state ^= rng_state >> 17;
        // rng_state ^= rng_state << 5;
        // u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF; // in [0,1]
        // p_dest[i+6] = u1-0.5f;

    }

    if (core_id == 0) {
        perf_bench_stop();
        perf_bench_read(&perf_end);
        perf_bench_diff(&perf_total, &perf_end, &perf_start);

        char label[100];
        snprintf(label, sizeof(label), "Perturb Eggroll seed=%u N=%u",
                seed, size);
        perf_bench_print(label, &perf_total);
    }
}

/* --------------------------- Update functions ---------------------------------- */

// void UpdateWeightsTriangle(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size) {
//     uint32_t rng_state = (seed * 1664525u) + 1013904223u;
//     float32_t sqrt6 = 2.44948974278f;
//     const float32_t scale = sqrt6; // sqrt(6): => Gaussian(0, 1) l2 norm.
//     for (uint32_t i = 0; i < size; i++) {
//         float32_t tr = TriangularSample(&rng_state);
//         pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * tr * scale;
//     }
// }

// void UpdateWeightsUniform(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size) {
//     uint32_t rng_state = (seed * 1664525u) + 1013904223u;
//     float32_t sqrt3 = 1.73205080757f;
//     const float32_t scale = sqrt3 * 2.0f; // factor 2: [-0.5,0.5] => [-1,1], sqrt(3): => variance 1
//     for (uint32_t i = 0; i < size; i++) {
//         float32_t u = UniformSample(&rng_state);
//         pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u * scale;
//     }
// }

// void UpdateWeightsGaussian(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size) {
//     uint32_t rng_state = (seed * 1664525u) + 1013904223u;
//     for (uint32_t i = 0; i < size; i++) {
//         float32_t u = GaussianSample(&rng_state);
//         pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u;
//     }
// }

// void UpdateWeightsRademacher(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size) {
//     RademacherRNG rng_state = { (seed * 1664525u) + 1013904223u, 0, 32 };
//     for (uint32_t i = 0; i < size; i++) {
//         float32_t u = RademacherSample(&rng_state);
//         pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u;
//     }
// }