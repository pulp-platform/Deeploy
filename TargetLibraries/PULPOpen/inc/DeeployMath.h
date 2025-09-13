/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_HEADER_
#define __DEEPLOY_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(AM_PART_APOLLO4B) | defined(DAM_PART_APOLLO3)
#include "am_bsp.h"
#include "am_mcu_apollo.h"
#include "am_util.h"
#endif

#include "DeeployBasicMath.h"

#endif // __DEEPLOY_MATH_HEADER_
