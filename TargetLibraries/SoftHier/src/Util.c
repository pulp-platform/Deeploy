/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySoftHierMath.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

int deeploy_log(const char *__restrict fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vprintf(fmt, args);
  va_end(args);
  return ret;
}

void *deeploy_malloc(const size_t size) {
  return (void *)flex_hbm_malloc(size);
}
void deeploy_free(void *const ptr) { flex_hbm_free(ptr); }
