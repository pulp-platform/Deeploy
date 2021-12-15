/* =====================================================================
 * Title:        DeeployMath.h
 * Description:
 *
 * $Date:        30.12.2021
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
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
#include <stdio.h>
#include <string.h>

#include "DeeployBasicMath.h"

#include "pmsis.h"

#include "kernel/RQiHardswish.h"
#include "kernel/RequantShift.h"
#include "kernel/UniformRequantShift.h"
#include "kernel/gemv.h"
#include "kernel/iRMSnorm.h"
#include "kernel/iSoftmax.h"

#endif // __DEEPLOY_MATH_HEADER_
