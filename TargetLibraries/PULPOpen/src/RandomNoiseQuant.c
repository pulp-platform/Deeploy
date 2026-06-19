/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

/* Work around LLVM assembler label collision bug at -O3 */
#pragma clang optimize off

/* --------- Quantized perturbation kernels (int8 and int32) --------- */

void ApplyPerturbQuantRademacher_CHW(int8_t *__restrict__ pweights,
                            int8_t *__restrict__ pweights_dest,
                            const int32_t *__restrict__ M, // Fixed-point multipliers
                            const int32_t S,             // Fixed-point shift
                            const uint32_t channel_width,
                            const uint32_t seed,
                            const uint32_t size,
                            const uint32_t start_offset)
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
            int32_t m_val = M[(start_offset + i) % channel_width];
            // Fixed-point multiplication: noise_q = round(r * M / 2^S)
            int32_t noise_q = (r * m_val + rounding) >> S;
            int32_t val = (int32_t)pweights[i] + noise_q;
            pweights_dest[i] = (int8_t)CLAMP(val, -127, 127); // Saturate to int8 range
            bits >>= 1;
            r = (bits & 1) ? 1 : -1;
            m_val = M[(start_offset + i + 1) % channel_width];
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
            int32_t m_val = M[(start_offset + i) % channel_width];
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
                                    const uint32_t size,
                                    const uint32_t start_offset)
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
            int32_t m_val = M[(start_offset + i + j) % channels];
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

        int32_t m_val = M[(start_offset + i) % channels];
        int32_t noise_q = (n_int * m_val + rounding) >> S;
        int32_t val = (int32_t)pweights[i] + noise_q;
        pweights_dest[i] = (int8_t)CLAMP(val, -127, 127);
    }
}

/* --------- int32 variants for bias perturbation --------- */

void ApplyPerturbQuantRademacher_i32(int32_t *__restrict__ pweights,
                            int32_t *__restrict__ pweights_dest,
                            const int32_t *__restrict__ M,
                            const int32_t S,
                            const uint32_t channel_width,
                            const uint32_t seed,
                            const uint32_t size,
                            const uint32_t start_offset)
{
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    const int32_t rounding = (S > 0) ? (1 << (S - 1)) : 0;

    uint32_t n_full_batches = size / 32;
    uint32_t leftover = size % 32;
    uint32_t i = 0;

    for (uint32_t batch = 0; batch < n_full_batches; batch++) {
        rng_state = Xorshift32(rng_state);
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < 32; b++, i++) {
            int32_t r = (bits & 1) ? 1 : -1;
            int32_t m_val = M[(start_offset + i) % channel_width];
            int32_t noise_q = (r * m_val + rounding) >> S;
            pweights_dest[i] = pweights[i] + noise_q;
            bits >>= 1;
        }
    }

    if (leftover > 0) {
        rng_state = Xorshift32(rng_state);
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < leftover; b++, i++) {
            int32_t r = (bits & 1) ? 1 : -1;
            int32_t m_val = M[(start_offset + i) % channel_width];
            int32_t noise_q = (r * m_val + rounding) >> S;
            pweights_dest[i] = pweights[i] + noise_q;
            bits >>= 1;
        }
    }
}

// Variant of ApplyPerturbQuantRademacher_CHW for weights stored output-channel
// first ([out, in], the GEMM kernel layout). The per-output-channel scale M is
// indexed by (position / channel_width) = the output-channel (row) index
// instead of (position % channel_width) = the inner (column) index. This keeps
// the per-channel scale in bounds for non-square GEMM weights (where in > out),
// for which the '%' kernel reads M out of bounds.
void ApplyPerturbQuantRademacher_CHW_ChannelFirst(int8_t *__restrict__ pweights,
                            int8_t *__restrict__ pweights_dest,
                            const int32_t *__restrict__ M,
                            const int32_t S,
                            const uint32_t channel_width,
                            const uint32_t seed,
                            const uint32_t size,
                            const uint32_t start_offset)
{
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    const int32_t rounding = (S > 0) ? (1 << (S - 1)) : 0;

    uint32_t n_full_batches = size / 32;
    uint32_t leftover = size % 32;
    uint32_t i = 0;

    for (uint32_t batch = 0; batch < n_full_batches; batch++) {
        rng_state = Xorshift32(rng_state);
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < 32; b++, i++) {
            int32_t r = (bits & 1) ? 1 : -1;
            int32_t m_val = M[(start_offset + i) / channel_width];
            int32_t noise_q = (r * m_val + rounding) >> S;
            int32_t val = (int32_t)pweights[i] + noise_q;
            pweights_dest[i] = (int8_t)CLAMP(val, -127, 127);
            bits >>= 1;
        }
    }

    if (leftover > 0) {
        rng_state = Xorshift32(rng_state);
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < leftover; b++, i++) {
            int32_t r = (bits & 1) ? 1 : -1;
            int32_t m_val = M[(start_offset + i) / channel_width];
            int32_t noise_q = (r * m_val + rounding) >> S;
            int32_t val = (int32_t)pweights[i] + noise_q;
            pweights_dest[i] = (int8_t)CLAMP(val, -127, 127);
            bits >>= 1;
        }
    }
}

void ApplyPerturbQuantUniform_i32(int32_t *__restrict__ pweights,
                                  int32_t *__restrict__ pweights_dest,
                                  const int32_t *__restrict__ M,
                                  const int32_t S,
                                  const uint32_t channels,
                                  const uint32_t seed,
                                  const uint32_t size,
                                  const uint32_t start_offset)
{
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    const int32_t rounding = (S > 0) ? (1 << (S - 1)) : 0;
    const uint8_t threshold = 255;

    uint32_t n_full_batches = size / 4;
    uint32_t leftover = size % 4;
    uint32_t i = 0;

    for (uint32_t batch = 0; batch < n_full_batches; batch++) {
        uint32_t random_word;
        int valid_word = 0;
        do {
            rng_state = Xorshift32(rng_state);
            random_word = rng_state;
            valid_word = 1;
            if (((random_word >> 0) & 0xFF) >= threshold ||
                ((random_word >> 8) & 0xFF) >= threshold ||
                ((random_word >> 16) & 0xFF) >= threshold ||
                ((random_word >> 24) & 0xFF) >= threshold) {
                valid_word = 0;
            }
        } while (!valid_word);

        int32_t n_int[4];
        n_int[0] = ((random_word >> 0) & 0xFF) % 3 - 1;
        n_int[1] = ((random_word >> 8) & 0xFF) % 3 - 1;
        n_int[2] = ((random_word >> 16) & 0xFF) % 3 - 1;
        n_int[3] = ((random_word >> 24) & 0xFF) % 3 - 1;

        for (int j = 0; j < 4; j++) {
            int32_t m_val = M[(start_offset + i + j) % channels];
            int32_t noise_q = (n_int[j] * m_val + rounding) >> S;
            pweights_dest[i + j] = pweights[i + j] + noise_q;
        }
        i += 4;
    }

    for (uint32_t b = 0; b < leftover; b++, i++) {
        uint8_t u;
        do {
            rng_state = Xorshift32(rng_state);
            u = rng_state & 0xFF;
        } while (u >= threshold);
        int32_t n_int = (u % 3) - 1;
        int32_t m_val = M[(start_offset + i) % channels];
        int32_t noise_q = (n_int * m_val + rounding) >> S;
        pweights_dest[i] = pweights[i] + noise_q;
    }
}
