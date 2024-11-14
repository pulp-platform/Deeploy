/*
 * ----------------------------------------------------------------------
 *
 * File: macros.h
 *
 * Last edited: 30.05.2024
 *
 * Copyright (C) 2024, ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
 *
 * ----------------------------------------------------------------------
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

#ifndef __DEEPLOY_MATH_MACROS_HEADER_
#define __DEEPLOY_MATH_MACROS_HEADER_

// #define log2(x)                                                 __builtin_pulp_fl1(x)
#define INT_LOG2(x) __builtin_ctz(x)

#endif //__DEEPLOY_MATH_MACROS_HEADER_
