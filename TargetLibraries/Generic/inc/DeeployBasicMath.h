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

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "macros.h"
#include "types.h"
#include "utils.h"

#include "kernel/AveragePool.h"
#include "kernel/BatchNorm.h"
#include "kernel/Ceil.h"
#include "kernel/Clip.h"
#include "kernel/ConvTranspose1d_fp32.h"
#include "kernel/Convolution.h"
#include "kernel/Cos.h"
#include "kernel/DWConvolution.h"
#include "kernel/Div.h"
#include "kernel/Erf.h"
#include "kernel/Exp.h"
#include "kernel/Expand.h"
#include "kernel/Floor.h"
#include "kernel/GELU.h"
#include "kernel/GatherElements.h"
#include "kernel/Gemm.h"
#include "kernel/GlobalAveragePool.h"
#include "kernel/GlobalMaxPool.h"
#include "kernel/GridSample.h"
#include "kernel/GroupNorm.h"
#include "kernel/HardSigmoid.h"
#include "kernel/HardSwish.h"
#include "kernel/InstanceNorm.h"
#include "kernel/Layernorm.h"
#include "kernel/Log.h"
#include "kernel/MatMul.h"
#include "kernel/MaxPool.h"
#include "kernel/Mod.h"
#include "kernel/Pow.h"
#include "kernel/RMSNorm.h"
#include "kernel/RQDiv.h"
#include "kernel/RQGELU.h"
#include "kernel/RQHardswish.h"
#include "kernel/Relu.h"
#include "kernel/RequantShift.h"
#include "kernel/Resize.h"
#include "kernel/ScatterND.h"
#include "kernel/Sigmoid.h"
#include "kernel/Sin.h"
#include "kernel/Softmax.h"
#include "kernel/Split.h"
#include "kernel/Sqrt.h"
#include "kernel/Swish.h"
#include "kernel/Tanh.h"
#include "kernel/Tile.h"
#include "kernel/TopK.h"
#include "kernel/Where.h"

#endif //__DEEPLOY_BASIC_MATH_HEADER_
