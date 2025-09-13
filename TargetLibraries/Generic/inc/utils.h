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
