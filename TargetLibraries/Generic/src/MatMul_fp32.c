/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void MatMul_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                           const float32_t *__restrict__ pSrcB,
                           float32_t *__restrict__ pDstY, uint32_t M,
                           uint32_t N, uint32_t O) {

  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < O; ++j) {
      float32_t sum = 0.0f;
      for (uint32_t k = 0; k < N; ++k) {
        sum += pSrcA[i * N + k] * pSrcB[k * O + j];
      }
      pDstY[i * O + j] = sum;
    }
  }
}

void MatMul_fp32_fp32_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                     const float32_t *__restrict__ pSrcB,
                                     float32_t *__restrict__ pDstY, uint32_t M,
                                     uint32_t N, uint32_t O) {
  uint32_t i, j, k;
  uint32_t O_block = O - (O % 7);

  for (i = 0; i < M; i++) {
    for (j = 0; j < O_block; j += 7) {
      float32_t sum0 = 0.0f;
      float32_t sum1 = 0.0f;
      float32_t sum2 = 0.0f;
      float32_t sum3 = 0.0f;
      float32_t sum4 = 0.0f;
      float32_t sum5 = 0.0f;
      float32_t sum6 = 0.0f;

      for (k = 0; k < N; k++) {
        float32_t a0 = pSrcA[i * N + k];

        float32_t b0 = pSrcB[k * O + (j + 0)];
        float32_t b1 = pSrcB[k * O + (j + 1)];
        float32_t b2 = pSrcB[k * O + (j + 2)];
        float32_t b3 = pSrcB[k * O + (j + 3)];
        float32_t b4 = pSrcB[k * O + (j + 4)];
        float32_t b5 = pSrcB[k * O + (j + 5)];
        float32_t b6 = pSrcB[k * O + (j + 6)];

        sum0 += a0 * b0;
        sum1 += a0 * b1;
        sum2 += a0 * b2;
        sum3 += a0 * b3;
        sum4 += a0 * b4;
        sum5 += a0 * b5;
        sum6 += a0 * b6;
      }

      pDstY[i * O + (j + 0)] = sum0;
      pDstY[i * O + (j + 1)] = sum1;
      pDstY[i * O + (j + 2)] = sum2;
      pDstY[i * O + (j + 3)] = sum3;
      pDstY[i * O + (j + 4)] = sum4;
      pDstY[i * O + (j + 5)] = sum5;
      pDstY[i * O + (j + 6)] = sum6;
    }

    for (j = O_block; j < O; j++) {
      float32_t sum = 0.0f;

      for (k = 0; k < N; k++) {
        float32_t a_val = pSrcA[i * N + k];
        float32_t b_val = pSrcB[k * O + j];
        float32_t prod = a_val * b_val;
        sum += prod;
      }

      pDstY[i * O + j] = sum;
    }
  }
}
