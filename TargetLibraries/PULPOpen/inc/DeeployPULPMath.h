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

#include "types.h"

#include "DeeployBasicMath.h"

#include "pmsis.h"

#include "kernel/Conv.h"
#include "kernel/GELU.h"
#include "kernel/Layernorm.h"
#include "kernel/Matmul.h"
#include "kernel/MaxPool.h"
#include "kernel/RQiHardswish.h"
#include "kernel/RequantShift.h"
#include "kernel/Softmax.h"
#include "kernel/UniformRequantShift.h"
#include "kernel/gemv.h"
#include "kernel/iRMSnorm.h"

#endif // __DEEPLOY_MATH_HEADER_
