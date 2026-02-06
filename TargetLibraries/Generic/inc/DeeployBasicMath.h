/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_HEADER_
#define __DEEPLOY_BASIC_MATH_HEADER_

// Define default empty wrapper for single core section
#ifndef BEGIN_SINGLE_CORE
#define BEGIN_SINGLE_CORE
#endif

#ifndef END_SINGLE_CORE
#define END_SINGLE_CORE
#endif

#ifndef SINGLE_CORE
#define SINGLE_CORE
#endif

#include <ctype.h>
#include <inttypes.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "macros.h"
#include "types.h"
#include "utils.h"

#include "kernel/BatchNorm.h"
#include "kernel/ConvTranspose1d_fp32.h"
#include "kernel/Convolution.h"
#include "kernel/DWConvolution.h"
#include "kernel/Div.h"
#include "kernel/GELU.h"
#include "kernel/Gemm.h"
#include "kernel/Hardswish.h"
#include "kernel/Layernorm.h"
#include "kernel/MatMul.h"
#include "kernel/MaxPool.h"
#include "kernel/Pow.h"
#include "kernel/RMSNorm.h"
#include "kernel/RQDiv.h"
#include "kernel/RQGELU.h"
#include "kernel/RQHardswish.h"
#include "kernel/Relu.h"
#include "kernel/RequantShift.h"
#include "kernel/Softmax.h"
#include "kernel/Sqrt.h"

#endif //__DEEPLOY_BASIC_MATH_HEADER_
