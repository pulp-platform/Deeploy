/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __MAGIA_ADD_KERNEL_HEADER_
#define __MAGIA_ADD_KERNEL_HEADER_


void MAGIA_Add(int8_t *pIn1, int8_t *pIn2, int32_t *pOut, uint32_t size, int32_t offset);

#endif // __MAGIA_ADD_KERNEL_HEADER_