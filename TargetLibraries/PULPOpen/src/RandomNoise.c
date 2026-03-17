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

    float32_t dn = ZIGGURAT_R;
    float32_t tn = dn;
    float32_t vn = ZIGGURAT_V;

    float32_t q = vn / expf(-0.5f * dn * dn);
    kn[0] = (uint32_t)((dn / q) * ZIGGURAT_INT32_MAX);
    kn[1] = 0;

    wn[0] = (float32_t)(q / ZIGGURAT_INT32_MAX);
    wn[ZIGGURAT_TABLE_SIZE - 1] = (float32_t)(dn / ZIGGURAT_INT32_MAX);

    fn[0] = 1.0f;
    fn[ZIGGURAT_TABLE_SIZE - 1] = (float32_t)expf(-0.5f * dn * dn);

    for (int i = ZIGGURAT_TABLE_SIZE - 2; i > 0; i--) {
        dn = sqrtf(-2.0 * logf(vn / dn + expf(-0.5f * dn * dn)));
        kn[i + 1] = (uint32_t)((dn / tn) * ZIGGURAT_INT32_MAX);
        tn = dn;
        fn[i] = (float32_t)expf(-0.5f * dn * dn);
        wn[i] = (float32_t)(dn / ZIGGURAT_INT32_MAX);
    }
    ziggurat_tables_initialized = 1;
}


// Convert uint32 -> float in (0,1], never 0
static inline float u32_to_u01_open(uint32_t u) {
    return ((float32_t)u + 1.0f) * 2.3283064365386963e-10f; // (u+1)/2^32
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
                x = -logf(u32_to_u01_open(*state)) / ZIGGURAT_R;
                *state = Xorshift32(*state);
                y = -logf(u32_to_u01_open(*state));
            } while (y + y < x * x);
            return (hz > 0) ? ZIGGURAT_R + x : -ZIGGURAT_R - x;
        }

        // Slow path: use a fresh uniform independent of hz
        float32_t x = (float32_t)hz * wn[iz];
        *state = Xorshift32(*state);
        float32_t u2 = (float32_t)(*state) * ZIGGURAT_INV_UINT32_MAX; // in [0,1]
        if (fn[iz] + u2 * (fn[iz - 1] - fn[iz]) < expf(-0.5f * x * x)) {
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
}

void ApplyUniformPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon)
{
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
}

void ApplyGaussianPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon) {

    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    if (dir == 0) {epsilon *= -1.0f;}
    for (uint32_t i = 0; i < size; i+=4) {
        // float32_t u = GaussianZigguratSample(&rng_state);
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        // mutate state to avoid same seed for u2.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        // Perform Box-Muller transform once to get two Gaussian samples
        float32_t mag = sqrtf(-2.0f * logf(u1));
        float32_t angle = 2.0f * PI_F * u2;

        float32_t z1 = mag * cosf(angle);
        float32_t z2 = mag * sinf(angle);
        pweights_dest[i] = pweights[i] + z1 * epsilon;
        pweights_dest[i + 1] = pweights[i + 1] + z2 * epsilon;


        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        // mutate state to avoid same seed for u2.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        // Perform Box-Muller transform once to get two Gaussian samples
        mag = sqrtf(-2.0f * logf(u1));
        angle = 2.0f * PI_F * u2;

        z1 = mag * cosf(angle);
        z2 = mag * sinf(angle);
        pweights_dest[i + 2] = pweights[i + 2] + z1 * epsilon;
        pweights_dest[i + 3] = pweights[i + 3] + z2 * epsilon;
    }
}

void ApplyRademacherPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon) {

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
        for (uint32_t b = 0; b < 32; b+=2, i+=2) {
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
}

void GenEggrollPerturbation(float32_t *__restrict__ p_dest,
                            uint32_t seed,
                            uint32_t size)
{
    // For compatibility with existing codegen templates. Currently maps to Rademacher noise.
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
}

void ApplyPerturbQuantRademacher_CHW(int8_t *__restrict__ pweights,
                            int8_t *__restrict__ pweights_dest,
                            const int32_t *__restrict__ M, // Fixed-point multipliers
                            const int32_t S,             // Fixed-point shift
                            const uint32_t channel_width,
                            const uint32_t seed,
                            const uint32_t size)
{
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    const int32_t rounding = (S > 0) ? (1 << (S - 1)) : 0;

    uint32_t n_full_batches = size / 32;
    uint32_t leftover = size % 32;
    uint32_t i = 0;

    // Process full batches
    for (uint32_t batch = 0; batch < n_full_batches; batch++) {
        rng_state = Xorshift32(rng_state);
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < 32; b+=2, i+=2) {
            int32_t r = (bits & 1) ? 1 : -1;
            int32_t m_val = M[i % channel_width];
            // Fixed-point multiplication: noise_q = round(r * M / 2^S)
            int32_t noise_q = (r * m_val + rounding) >> S;
            int32_t val = (int32_t)pweights[i] + noise_q;
            pweights_dest[i] = (int8_t)CLAMP(val, -127, 127); // Saturate to int8 range
            bits >>= 1;
            r = (bits & 1) ? 1 : -1;
            m_val = M[i % channel_width];
            // Fixed-point multiplication: noise_q = round(r * M / 2^S)
            noise_q = (r * m_val + rounding) >> S;
            val = (int32_t)pweights[i+1] + noise_q;
            pweights_dest[i+1] = (int8_t)CLAMP(val, -127, 127); // Saturate to int8 range
            bits >>= 1;
        }
    }

    // Process leftover elements
    if (leftover > 0) {
        rng_state = Xorshift32(rng_state);
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < leftover; b++, i++) {
            int32_t r = (bits & 1) ? 1 : -1;
            int32_t m_val = M[i % channel_width];
            int32_t noise_q = (r * m_val + rounding) >> S;
            int32_t val = (int32_t)pweights[i] + noise_q;
            pweights_dest[i] = (int8_t)CLAMP(val, -127, 127);
            bits >>= 1;
        }
    }
}


void ApplyPerturbQuantUniform_NHWC(int8_t *__restrict__ pweights,
                                    int8_t *__restrict__ pweights_dest,
                                    const int32_t *__restrict__ M, // Fixed-point multipliers
                                    const int32_t S,             // Fixed-point shift
                                    const uint32_t channels,
                                    const uint32_t seed,
                                    const uint32_t size)
{
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    const int32_t rounding = (S > 0) ? (1 << (S - 1)) : 0;
    const uint8_t threshold = 255; // Reject if uint8_t value is >= 255

    uint32_t n_full_batches = size / 4;
    uint32_t leftover = size % 4;
    uint32_t i = 0;

    for (uint32_t batch = 0; batch < n_full_batches; batch++) {
        uint32_t random_word;
        int valid_word = 0;

        // Loop until we get a 32-bit word where all 4 bytes are valid.
        do {
            rng_state = Xorshift32(rng_state);
            random_word = rng_state;
            valid_word = 1; // Assume it's valid initially
            if (((random_word >> 0) & 0xFF) >= threshold ||
                ((random_word >> 8) & 0xFF) >= threshold ||
                ((random_word >> 16) & 0xFF) >= threshold ||
                ((random_word >> 24) & 0xFF) >= threshold) {
                valid_word = 0; // Invalidate if any byte is bad
            }
        } while (!valid_word);

        // Now that we have a valid word, extract the 4 samples.
        int32_t n_int[4];
        n_int[0] = ((random_word >> 0) & 0xFF) % 3 - 1;
        n_int[1] = ((random_word >> 8) & 0xFF) % 3 - 1;
        n_int[2] = ((random_word >> 16) & 0xFF) % 3 - 1;
        n_int[3] = ((random_word >> 24) & 0xFF) % 3 - 1;

        // Apply the 4 samples
        for (int j = 0; j < 4; j++) {
            int32_t m_val = M[(i + j) % channels];
            int32_t noise_q = (n_int[j] * m_val + rounding) >> S;
            int32_t val = (int32_t)pweights[i + j] + noise_q;
            pweights_dest[i + j] = (int8_t)CLAMP(val, -127, 127);
        }
        i += 4;
    }
    // Process leftover elements
    for (uint32_t b = 0; b < leftover; b++, i++) {
        uint8_t u;
        do {
            rng_state = Xorshift32(rng_state);
            // Just use the lowest 8 bits for simplicity
            u = rng_state & 0xFF;
        } while (u >= threshold);
        int32_t n_int = (u % 3) - 1;

        int32_t m_val = M[i % channels];
        int32_t noise_q = (n_int * m_val + rounding) >> S;
        int32_t val = (int32_t)pweights[i] + noise_q;
        pweights_dest[i] = (int8_t)CLAMP(val, -127, 127);
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