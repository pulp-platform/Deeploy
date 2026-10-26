/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CARFIELD_KERNEL_CONV_H
#define CARFIELD_KERNEL_CONV_H

#include "Types.h"
#include <stdint.h>

void Carfield_Conv(float32_t *input, float32_t *weight, float32_t *bias,
                   float32_t *output, int32_t F, int32_t C, int32_t H_in,
                   int32_t W_in, int32_t P, int32_t Q, int32_t core_id,
                   int32_t numThreads);

#endif // CARFIELD_KERNEL_CONV_H
