/* =====================================================================
 * Title:        util.h
 * Description:
 *
 * Date:         06.12.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DEEPLOY_BASIC_MATH_UTIL_HEADER_
#define __DEEPLOY_BASIC_MATH_UTIL_HEADER_

int deeploy_log(const char *__restrict fmt, ...)
    __attribute__((__format__(__printf__, 1, 2)));
void *deeploy_malloc(const size_t size);
void deeploy_free(void *const ptr);

void PrintMatrix_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t N,
                         uint32_t C, uint32_t H, uint32_t W, int32_t offset);

void PrintMatrix_s8_NHWC(int8_t const *__restrict__ pSrcA, uint32_t N,
                         uint32_t C, uint32_t H, uint32_t W, int32_t offset);

void PrintMatrix_s16_NCHW(int16_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, int32_t offset);

void PrintMatrix_s16_NHWC(int16_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, int32_t offset);

void PrintMatrix_s32_NCHW(int32_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, int32_t offset);

void PrintMatrix_s32_NHWC(int32_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, int32_t offset);

void PrintMatrix_u8_NCHW(uint8_t const *__restrict__ pSrcA, uint32_t N,
                         uint32_t C, uint32_t H, uint32_t W, uint32_t offset);

void PrintMatrix_u8_NHWC(uint8_t const *__restrict__ pSrcA, uint32_t N,
                         uint32_t C, uint32_t H, uint32_t W, uint32_t offset);

void PrintMatrix_u16_NCHW(uint16_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset);

void PrintMatrix_u16_NHWC(uint16_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset);

void PrintMatrix_u32_NCHW(uint32_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset);

void PrintMatrix_u32_NHWC(uint32_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset);

void PrintArray_s8(int8_t const *__restrict__ pSrcA, uint32_t N,
                   int32_t offset);

void PrintArray_s16(int16_t const *__restrict__ pSrcA, uint32_t N,
                    int32_t offset);

void PrintArray_s32(int32_t const *__restrict__ pSrcA, uint32_t N,
                    int32_t offset);

void PrintArray_u8(uint8_t const *__restrict__ pSrcA, uint32_t N,
                   uint32_t offset);

void PrintArray_u16(uint16_t const *__restrict__ pSrcA, uint32_t N,
                    uint32_t offset);

void PrintArray_u32(uint32_t const *__restrict__ pSrcA, uint32_t N,
                    uint32_t offset);

#endif //__DEEPLOY_BASIC_MATH_UTIL_HEADER_
