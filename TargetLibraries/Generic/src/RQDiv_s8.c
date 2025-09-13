/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void RQDiv_s32_s8(int32_t *data_in_nom, int32_t *data_in_denom,
                  int32_t size_nom, int32_t __attribute__((unused)) size_denom,
                  int32_t nomStep, int32_t denomStep, int8_t *data_out,
                  int32_t Delta, int32_t eps, int32_t eta, int32_t requant_mul,
                  int32_t requant_add, int32_t requant_shift) {

  int32_t innerMostIter = denomStep;
  int32_t secondIter = nomStep / innerMostIter;
  int32_t thirdIter = size_nom / secondIter;
  int64_t nom;
  int32_t sgnNom = 0;
  int64_t denom;
  int32_t y, intermediate;

  for (int i = 0; i < thirdIter; i++) {
    for (int k = 0; k < innerMostIter; k++) {
      denom = data_in_denom[i * innerMostIter + k];
      denom = ((eta * denom) + eps);
      for (int j = 0; j < secondIter; j++) {
        nom =
            data_in_nom[i * secondIter * innerMostIter + j * innerMostIter + k];
        nom = (Delta * eta * nom);
        sgnNom = (nom >= 0) - (nom < 0);
        y = (int32_t)((nom + sgnNom * (denom >> 1)) / denom);
        intermediate = (int32_t)(y)*requant_mul + requant_add;
        intermediate =
            ((intermediate + ((1 << (requant_shift - 1)))) >> requant_shift);
        *data_out++ = (int8_t)CLAMP(intermediate, -128, 127);
      }
    }
  }
}
