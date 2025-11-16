/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// Overwrite weak function from DeeployBasicLibs
int deeploy_log(const char *__restrict fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret;

#if defined(AM_PART_APOLLO4B) | defined(DAM_PART_APOLLO3)
  ret = am_util_stdio_vprintf(fmt, args);
#else
  ret = vprintf(fmt, args);
#endif

  va_end(args);
  return ret;
}

void *deeploy_malloc(const size_t size) { return malloc(size); }

void deeploy_free(void *const ptr) { free(ptr); }
