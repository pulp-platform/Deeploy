/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
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

#include "types.h"

#define BEGIN_SINGLE_CORE if (pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 0)

#include "DeeployBasicMath.h"

#include "pmsis.h"

#include "kernel/Conv.h"
#include "kernel/GELU.h"
#include "kernel/Layernorm.h"
#include "kernel/Matmul.h"
#include "kernel/MaxPool.h"
#include "kernel/RQiHardswish.h"
#include "kernel/RequantShift.h"
#include "kernel/Softmax.h"
#include "kernel/UniformRequantShift.h"
#include "kernel/gemm.h"
#include "kernel/gemv.h"
#include "kernel/iRMSnorm.h"

#define LOG2(x) (__builtin_pulp_fl1(x))

#endif // __DEEPLOY_MATH_HEADER_
