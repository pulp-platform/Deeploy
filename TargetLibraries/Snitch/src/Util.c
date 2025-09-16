/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// Provide implentation for extern functions decalred in DeeployBasicLibs
int deeploy_log(const char *__restrict fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vprintf_(fmt, args);
  va_end(args);
  return ret;
}

void *deeploy_malloc(const size_t size) { return snrt_l1alloc(size); }

void deeploy_free(void *const __attribute__((unused)) ptr) {}
