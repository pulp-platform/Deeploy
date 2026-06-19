/*
 * SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_HEADER_
#define __DEEPLOY_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BEGIN_SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0)

// LOG2 via PULP-ISA find-last-one builtin (same as DeeployPULPMath.h)
#define LOG2(x) (__builtin_pulp_fl1(x))

#include "DeeployBasicMath.h"

#include "dory_dma.h"
#include "dory_mem.h"

#include "pmsis.h"

// PULPOpen kernel function declarations.
// DeeployPULPMath.h is blocked by the __DEEPLOY_MATH_HEADER_ guard above, but
// float32_t / uint32_t etc. are already defined via DeeployBasicMath.h.
#include "kernel/Conv.h"
#include "kernel/GELU.h"
#include "kernel/Layernorm.h"
#include "kernel/Matmul.h"
#include "kernel/MaxPool.h"
#include "kernel/Relu.h"
#include "kernel/Softmax.h"
#include "kernel/gemm.h"
#include "kernel/gemv.h"

#endif // __DEEPLOY_MATH_HEADER_
