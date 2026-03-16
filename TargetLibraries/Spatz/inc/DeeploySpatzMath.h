/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_SPATZ_MATH_HEADER_
#define __DEEPLOY_SPATZ_MATH_HEADER_

#include <stdint.h>
#include <stdbool.h>

#include "DeeployBasicMath.h"
#include "snrt.h"

#define BEGIN_SINGLE_CORE if (core_id == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (core_id == 0)

#endif // __DEEPLOY_SPATZ_MATH_HEADER_
