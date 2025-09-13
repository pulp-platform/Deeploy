/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_UTIL_HEADER_
#define __DEEPLOY_BASIC_MATH_UTIL_HEADER_

int deeploy_log(const char *__restrict fmt, ...)
    __attribute__((__format__(__printf__, 1, 2)));
void *deeploy_malloc(const size_t size);
void deeploy_free(void *const ptr);

#endif //__DEEPLOY_BASIC_MATH_UTIL_HEADER_
