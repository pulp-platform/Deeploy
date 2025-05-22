/* =====================================================================
 * Title:        DeeployBasicMath.h
 * Description:
 *
 * Date:         14.12.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese, ETH Zurich
 * - Victor Jung, ETH Zurich
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
#include "util.h"

#endif //__DEEPLOY_BASIC_MATH_HEADER_
