/* =====================================================================
 * Title:        DeeploySnitchMath.h
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
 * - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
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
#include "macros.h"

#include "DeeployBasicMath.h"

#include "snrt.h"

#include "kernel/Gemm.h"
#include "kernel/MatMul.h"
#include "kernel/RQGemm.h"
#include "kernel/RQMatMul.h"
#include "kernel/Softmax.h"
#include "kernel/UniformRequantShift.h"

#endif //__DEEPLOY_MATH_HEADER_
