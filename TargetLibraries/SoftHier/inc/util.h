/* =====================================================================
 * Title:        util.h
 * Description:
 *
 * Date:         07.06.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Bowen Wang <bowwang@iis.ee.ethz.ch>, ETH Zurich
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

#ifndef __DEEPLOY_BASIC_MATH_UTIL_HEADER_
#define __DEEPLOY_BASIC_MATH_UTIL_HEADER_

int deeploy_log(const char *__restrict fmt, ...)
    __attribute__((__format__(__printf__, 1, 2)));
void *deeploy_malloc(const size_t size);
void deeploy_free(void *const ptr);

#endif //__DEEPLOY_BASIC_MATH_UTIL_HEADER_
