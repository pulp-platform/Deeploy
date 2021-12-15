/* =====================================================================
 * Title:        dory_mem.c
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

#include "dory_mem.h"
#include "bsp/bsp.h"
#include "bsp/flash.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/ram.h"
#include "pmsis.h"

#ifdef USE_HYPERFLASH
#include "bsp/flash/hyperflash.h"
typedef struct pi_hyperflash_conf flash_conf_t;
#define flash_conf_init(conf) pi_hyperflash_conf_init(conf)
#elif defined USE_SPIFLASH
#include "bsp/flash/spiflash.h"
typedef struct pi_spiflash_conf flash_conf_t;
#define flash_conf_init(conf) pi_spiflash_conf_init(conf)
#elif defined USE_MRAM
typedef struct pi_mram_conf flash_conf_t;
#define flash_conf_init(conf) pi_mram_conf_init(conf)
#else
typedef struct pi_default_flash_conf flash_conf_t;
#define flash_conf_init(conf) pi_default_flash_conf_init(conf)
#endif

#ifdef USE_HYPERRAM
#include "bsp/ram/hyperram.h"
typedef struct pi_hyperram_conf ram_conf_t;
#define ram_conf_init(conf) pi_hyperram_conf_init(conf)
#else
typedef struct pi_default_ram_conf ram_conf_t;
#define ram_conf_init(conf) pi_default_ram_conf_init(conf)
#endif

#define BUFFER_SIZE 128
static uint8_t buffer[BUFFER_SIZE];

static struct pi_device flash;
static flash_conf_t flash_conf;

static struct pi_device fs;
static struct pi_readfs_conf fs_conf;

struct pi_device ram;
static ram_conf_t ram_conf;

void open_fs() {
  // SCHEREMO: Fix FS
  // Open filesystem on flash.
  pi_readfs_conf_init(&fs_conf);
  fs_conf.fs.flash = &flash;
  pi_open_from_conf(&fs, &fs_conf);
  if (pi_fs_mount(&fs)) {
    printf("ERROR: Cannot mount filesystem! Exiting...\n");
    pmsis_exit(-2);
  }
}

void mem_init() {
  flash_conf_init(&flash_conf);
  pi_open_from_conf(&flash, &flash_conf);
  if (pi_flash_open(&flash)) {
    printf("ERROR: Cannot open flash! Exiting...\n");
    pmsis_exit(-1);
  }

  ram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  if (pi_ram_open(&ram)) {
    printf("ERROR: Cannot open ram! Exiting...\n");
    pmsis_exit(-3);
  }
}

struct pi_device *get_ram_ptr() { return &ram; }

void *ram_malloc(size_t size) {
  void *ptr = NULL;
  pi_ram_alloc(&ram, &ptr, size);
  return ptr;
}

void ram_free(void *ptr, size_t size) { pi_ram_free(&ram, ptr, size); }

void ram_read(void *dest, void *src, const size_t size) {
  pi_ram_read(&ram, src, dest, size);
}

void ram_write(void *dest, void *src, const size_t size) {
  pi_ram_write(&ram, dest, src, size);
}

void *cl_ram_malloc(size_t size) {
  int addr;
  pi_cl_ram_req_t req;
  pi_cl_ram_alloc(&ram, size, &req);
  pi_cl_ram_alloc_wait(&req, &addr);
  return (void *)addr;
}

void cl_ram_free(void *ptr, size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_free(&ram, ptr, size, &req);
  pi_cl_ram_free_wait(&req);
}

void cl_ram_read(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_read(&ram, src, dest, size, &req);
  pi_cl_ram_read_wait(&req);
}

void cl_ram_write(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_write(&ram, dest, src, size, &req);
  pi_cl_ram_write_wait(&req);
}

size_t load_file_to_ram(const void *dest, const char *filename) {
  pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
  if (fd == NULL) {
    printf("ERROR: Cannot open file %s! Exiting...", filename);
    pmsis_exit(-4);
  }

  size_t size = fd->size;
  size_t load_size = 0;
  size_t remaining_size = size;

  size_t offset = 0;
  do {

    remaining_size = size - offset;
    load_size = BUFFER_SIZE < remaining_size ? BUFFER_SIZE : remaining_size;

    pi_cl_fs_req_t req;
    pi_cl_fs_read(fd, buffer, load_size, &req);
    pi_cl_fs_wait(&req);
    cl_ram_write(dest + offset, buffer, load_size);
    offset += load_size;
  } while (offset < size);

  return offset;
}
