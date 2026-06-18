/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */
 
#ifndef __DEEPLOY_MATH_RANDOMNOISE_KERNEL_HEADER_
#define __DEEPLOY_MATH_RANDOMNOISE_KERNEL_HEADER_

#include "DeeployPULPMath.h"


#define PI_F 3.14159265358979323846f

#define ZIGGURAT_TABLE_SIZE 128
#define ZIGGURAT_R 3.442619855899f
#define ZIGGURAT_UINT32_MAX 4294967296.0f
#define ZIGGURAT_INV_UINT32_MAX 2.3283064365386963e-10f
#define ZIGGURAT_INT32_MAX  2147483648.0f      // 2^31
#define ZIGGURAT_INV_INT32_MAX (1.0f / 2147483648.0f)

#define ZIGGURAT_V 9.91256303526217e-3f

static uint32_t kn[ZIGGURAT_TABLE_SIZE];
static float32_t wn[ZIGGURAT_TABLE_SIZE];
static float32_t fn[ZIGGURAT_TABLE_SIZE];
static int32_t ziggurat_tables_initialized = 0;


typedef struct {
    uint32_t state;
    uint32_t bits;
    uint32_t bitpos;
} RademacherRNG;

// Sample from Unifom distribution U[-0.5,0.5]
float32_t UniformSample(uint32_t *state);
// Sample from triangular distribution Tr[-1, 1]
float32_t TriangularSample(uint32_t *state);
float32_t GaussianSample(uint32_t *state);
float32_t RademacherSample(RademacherRNG *rng);
uint32_t Xorshift32(uint32_t state);
void build_ziggurat_tables();
float32_t GaussianZigguratSample(uint32_t *state);

void RademacherRNG_init(RademacherRNG *rng, uint32_t seed);

// Applies triangular perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyTrianglePerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon);

// Applies uniform perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyUniformPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon);
                        
// Applies uniform perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyGaussianPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon);
                     
// Applies uniform perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyRademacherPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon);

// Rademacher perturbation using only the LSB of the Xorshift state, advancing the
// PRNG once per element (same stepping as ApplyUniformPerturbation).
void ApplySequentialRademacherPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon);

static inline float u32_to_u01_open(uint32_t u);

void ApplyPerturbQuantRademacher_CHW(int8_t *__restrict__ pweights,
                            int8_t *__restrict__ pweights_dest,
                            const int32_t *__restrict__ M, // Fixed-point multipliers
                            const int32_t S,             // Fixed-point shift
                            const uint32_t channel_width,
                            const uint32_t seed,
                            const uint32_t size,
                            const uint32_t start_offset);


void ApplyPerturbQuantUniform_NHWC(int8_t *__restrict__ pweights,
                                    int8_t *__restrict__ pweights_dest,
                                    const int32_t *__restrict__ M, // Fixed-point multipliers
                                    const int32_t S,             // Fixed-point shift
                                    const uint32_t channels,
                                    const uint32_t seed,
                                    const uint32_t size,
                                    const uint32_t start_offset);

void ApplyPerturbQuantRademacher_i32(int32_t *__restrict__ pweights,
                            int32_t *__restrict__ pweights_dest,
                            const int32_t *__restrict__ M,
                            const int32_t S,
                            const uint32_t channel_width,
                            const uint32_t seed,
                            const uint32_t size,
                            const uint32_t start_offset);


void ApplyPerturbQuantUniform_i32(int32_t *__restrict__ pweights,
                                  int32_t *__restrict__ pweights_dest,
                                  const int32_t *__restrict__ M,
                                  const int32_t S,
                                  const uint32_t channels,
                                  const uint32_t seed,
                                  const uint32_t size,
                                  const uint32_t start_offset);
// Updates the weights in place according to the MeZO update rule with triangular noise.
// Only supports qMeZO with q = 1 for now.
// void UpdateWeightsTriangle(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size);

// // Updates the weights in place according to the MeZO update rule with uniform noise.
// // Only supports qMeZO with q = 1 for now.
// void UpdateWeightsUniform(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size);

// void UpdateWeightsGaussian(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size);

// void UpdateWeightsRademacher(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size);


void GenEggrollPerturbation(float32_t *__restrict__ p_dest,
                            uint32_t seed,
                            uint32_t size);

#endif //__DEEPLOY_MATH_RANDOMNOISE_KERNEL_HEADER_