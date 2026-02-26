/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */
 
#ifndef __DEEPLOY_MATH_RANDOMNOISE_KERNEL_HEADER_
#define __DEEPLOY_MATH_RANDOMNOISE_KERNEL_HEADER_

#include "DeeployPULPMath.h"


#define PI_F 3.14159265358979323846f


typedef struct {
    uint32_t state;
    uint32_t bits;
    int bitpos;
} RademacherRNG;

// Sample from Unifom distribution U[-0.5,0.5]
float32_t UniformSample(uint32_t *state);
// Sample from triangular distribution Tr[-1, 1]
float32_t TriangularSample(uint32_t *state);
float32_t GaussianSample(uint32_t *state);
float32_t RademacherSample(RademacherRNG *rng);

void RademacherRNG_init(RademacherRNG *rng, uint32_t seed);

// Applies triangular perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyTriangularPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            float32_t epsilon,
                            uint32_t dir,
                            uint32_t size);

// Applies uniform perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyUniformPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            float32_t epsilon,
                            uint32_t dir,
                            uint32_t size);
                        
// Applies uniform perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyGaussianPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            float32_t epsilon,
                            uint32_t dir,
                            uint32_t size);
                     
// Applies uniform perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyRademacherPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            float32_t epsilon,
                            uint32_t dir,
                            uint32_t size);


// Updates the weights in place according to the MeZO update rule with triangular noise.
// Only supports qMeZO with q = 1 for now.
void UpdateWeightsTriangle(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size);

// Updates the weights in place according to the MeZO update rule with uniform noise.
// Only supports qMeZO with q = 1 for now.
void UpdateWeightsUniform(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size);

void UpdateWeightsGaussian(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size);

void UpdateWeightsRademacher(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size);


void GenEggrollPerturbation(float32_t * pdest,
                        uint32_t seed,
                        float32_t epsilon,
                        uint32_t size);

/* Xorshift32 implementation. Most basic software PRNG*/
uint32_t Xorshift32(uint32_t state);

#endif //__DEEPLOY_MATH_RANDOMNOISE_KERNEL_HEADER_