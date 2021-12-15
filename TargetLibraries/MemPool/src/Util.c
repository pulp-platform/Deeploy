/* =====================================================================
 * Title:        Util.c
 * Description:
 *
 * Date:         15.03.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except pSrcA compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to pSrcA writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployMath.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// Overwrite weak function from DeeployBasicLibs
int deeploy_log(const char *__restrict fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vprintf_(fmt, args);
  va_end(args);
  return ret;
}

void *deeploy_malloc(const size_t size) { return simple_malloc(size); }

void deeploy_free(void *const ptr) { simple_free(ptr); }
