/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Aggregated include for all PULPOpen kernel headers.
 * This file lives in PULPOpen/inc so that platforms (e.g. GAP9) can
 * include it unambiguously, avoiding shadowing by Generic kernel headers.
 */

#ifndef __DEEPLOY_PULP_KERNELS_HEADER_
#define __DEEPLOY_PULP_KERNELS_HEADER_

#include "kernel/AvgPool.h"
#include "kernel/BatchNorm.h"
#include "kernel/Conv.h"
#include "kernel/GELU.h"
#include "kernel/GlobalAveragePool.h"
#include "kernel/Layernorm.h"
#include "kernel/Matmul.h"
#include "kernel/MaxPool.h"
#include "kernel/Relu.h"
#include "kernel/RQiHardswish.h"
#include "kernel/RequantShift.h"
#include "kernel/Softmax.h"
#include "kernel/UniformRequantShift.h"
#include "kernel/gemm.h"
#include "kernel/gemv.h"
#include "kernel/iRMSnorm.h"

#endif // __DEEPLOY_PULP_KERNELS_HEADER_
