/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
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
#include "ITA.h"
#include "constants.h"
#include "macros.h"

#include "DeeployBasicMath.h"

#include "builtins.h"
#include "dma.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "kernel/Convolution.h"
#include "kernel/DWConvolution.h"
#include "kernel/Gemm.h"
#include "kernel/MHSA.h"
#include "kernel/MatMul.h"
#include "kernel/MaxPool.h"
#include "kernel/RQGemm.h"
#include "kernel/RQMatMul.h"
#include "kernel/RequantShift.h"
#include "kernel/Softmax.h"

#endif //__DEEPLOY_MATH_HEADER_
