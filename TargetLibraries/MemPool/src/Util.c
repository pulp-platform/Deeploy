/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
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
