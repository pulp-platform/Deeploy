/* =====================================================================
 * Title:        dory_mem.h
 * Description:
 *
 * $Date:        12.12.2023
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
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

#ifndef __MEM_H__
#define __MEM_H__

#include <stddef.h>

extern struct pi_device ram;

void open_fs();
void mem_init();
struct pi_device *get_ram_ptr();
void *ram_malloc(size_t size);
void ram_free(void *ptr, size_t size);
void ram_read(void *dest, void *src, size_t size);
void ram_write(void *dest, void *src, size_t size);
void *cl_ram_malloc(size_t size);
void cl_ram_free(void *ptr, size_t size);
void cl_ram_read(void *dest, void *src, size_t size);
void cl_ram_write(void *dest, void *src, size_t size);
size_t load_file_to_ram(const void *dest, const char *filename);
size_t load_file_to_local(const void *dest, const char *filename);

#endif // __MEM_H__
