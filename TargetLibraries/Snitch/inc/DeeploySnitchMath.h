/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_HEADER_
#define __DEEPLOY_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define BEGIN_SINGLE_CORE if (snrt_cluster_core_idx() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (snrt_cluster_core_idx() == 0)

#include "CycleCounter.h"
#include "macros.h"

#include "DeeployBasicMath.h"

#include "snrt.h"

// Packed pair of fp32 lanes (8 bytes), matching the 64-bit SSR/FPU register
// width used by the vectorized (vfXXX.s) Snitch kernels.
typedef float v2f32 __attribute__((vector_size(8)));

#include "kernel/Add.h"
#include "kernel/Div.h"
#include "kernel/Gemm.h"
#include "kernel/Gemm_fp32.h"
#include "kernel/HardSwish.h"
#include "kernel/MatMul.h"
#include "kernel/Mul.h"
#include "kernel/RMSNrom.h"
#include "kernel/RQGemm.h"
#include "kernel/RQMatMul.h"
#include "kernel/Softmax.h"
#include "kernel/UniformRequantShift.h"
#include "kernel/iNoNorm.h"

#include "dmaStruct.h"

#endif //__DEEPLOY_MATH_HEADER_
