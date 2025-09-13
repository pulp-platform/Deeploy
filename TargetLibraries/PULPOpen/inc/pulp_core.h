/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_PULPCORE_HEADER_
#define __DEEPLOY_MATH_PULPCORE_HEADER_

#define BEGIN_SINGLE_CORE if (pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 0)

#endif //__DEEPLOY_MATH_PULPCORE_HEADER_
