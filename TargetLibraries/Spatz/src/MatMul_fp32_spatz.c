// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include "DeeploySpatzMath.h"

void matmul(float *c, const float *a, const float *b, const unsigned int M,
            const unsigned int N, const unsigned int P);

void Spatz_MatMul_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                                 const float32_t *__restrict__ pSrcB,
                                 float32_t *__restrict__ pDstY, uint32_t M,
                                 uint32_t N, uint32_t O) {
	// defined in ${SPATZ_HOME}/sw/spatzBenchmarks/sp-fmatmul/kernel/sp-fmatmul.c
  matmul(pDstY, pSrcA, pSrcB, M, N, O);
}
