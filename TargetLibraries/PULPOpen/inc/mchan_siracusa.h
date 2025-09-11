/* ----------------------------------------------------------------------
#
# File: mchan_siracusa.h
#
# Last edited: 11.09.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Luka Macan, luka.macan@unibo.it, University of Bologna
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

// Default mchan base address
#ifndef MCHAN_BASE_ADDR
#define MCHAN_BASE_ADDR (ARCHI_MCHAN_DEMUX_ADDR) // CLUSTER_MCHAN_ADDR
#endif

// Default mchan await mode
#if !defined(MCHAN_EVENT) && !defined(MCHAN_POLLED)
#define MCHAN_EVENT
#endif

#ifdef MCHAN_EVENT
#define MCHAN_EVENT_BIT (ARCHI_CL_EVT_DMA0) // 8
#endif

#include "mchan_v7.h"
