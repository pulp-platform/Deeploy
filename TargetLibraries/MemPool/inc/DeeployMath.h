/* =====================================================================
 * Title:        DeeployMath.h
 * Description:
 *
 * Date:         29.11.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Samuel Riedel, ETH Zurich
 * - Sergio Mazzola, ETH Zurich
 * - Philip Wiese, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
