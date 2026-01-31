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

#define BEGIN_SINGLE_CORE if (core_id == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (core_id == 0)

#include "CycleCounter.h"
#include "macros.h"

#include "DeeployBasicMath.h"

#include "snrt.h"

#include "kernel/Add.h"
#include "kernel/Div.h"
#include "kernel/Gemm.h"
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
