/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dory_mem.h"
#include "bsp/bsp.h"
// #include "bsp/flash.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
// #include "bsp/ram.h"
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

#define BUFFER_SIZE 2048 // 128
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
  pi_ram_alloc(&ram, (uint32_t *)&ptr, size);
  return ptr;
}

void ram_free(void *ptr, size_t size) {
  pi_ram_free(&ram, (uint32_t)ptr, size);
}

void ram_read(void *dest, void *src, const size_t size) {
  pi_ram_read(&ram, (uint32_t)src, dest, size);
}

void ram_write(void *dest, void *src, const size_t size) {
  pi_ram_write(&ram, (uint32_t)dest, src, size);
}

void *cl_ram_malloc(size_t size) {
  uint32_t addr;
  pi_cl_ram_alloc_req_t req;
  pi_cl_ram_alloc(&ram, size, &req);
  pi_cl_ram_alloc_wait(&req, &addr);
  return (void *)addr;
}

void cl_ram_free(void *ptr, size_t size) {
  pi_cl_ram_free_req_t req;
  pi_cl_ram_free(&ram, (uint32_t)ptr, size, &req);
  pi_cl_ram_free_wait(&req);
}

void cl_ram_read(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_read(&ram, (uint32_t)src, dest, size, &req);
  pi_cl_ram_read_wait(&req);
}

void cl_ram_write(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_write(&ram, (uint32_t)dest, src, size, &req);
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
  float fb0[8];     // first 32 bytes as read from flash (file start)
  int captured = 0;
  do {

    remaining_size = size - offset;
    load_size = BUFFER_SIZE < remaining_size ? BUFFER_SIZE : remaining_size;

    pi_cl_fs_req_t req;
    pi_cl_fs_read(fd, buffer, load_size, &req);
    pi_cl_fs_wait(&req);
    if (!captured && load_size >= 32) {
      for (int i = 0; i < 8; i++)
        fb0[i] = ((const float *)buffer)[i];
      captured = 1;
    }
    cl_ram_write(dest + offset, buffer, load_size);
    offset += load_size;
  } while (offset < size);

  // ---- DEBUG: discriminate flash-read-garbage vs L3-write-not-persisted ----
  // fb0 = file's first 8 floats as read from flash; l3rb = same bytes read back
  // from L3 (`dest`) after the write. If flash is sane but l3rb is garbage =>
  // the L3 write didn't persist at this address on silicon. If flash is already
  // garbage => readfs returned garbage for this (2nd/3rd) file.
  {
    float rb[8];
    cl_ram_read(rb, (void *)dest, 32);
    printf("[LOADDBG] %s dest=0x%08x size=%u flash:", filename, (unsigned)(uint32_t)dest, (unsigned)size);
    for (int i = 0; i < 8; i++)
      printf(" %.5f", fb0[i]);
    printf("  l3rb:");
    for (int i = 0; i < 8; i++)
      printf(" %.5f", rb[i]);
    printf("\r\n");
  }

  return offset;
}

// DEBUG: scan a readfs file in chunks via the normal FS read and find the byte
// offset where data turns from sane floats (|v|<100) to garbage (nan/inf/huge).
// Run on 0.hex (input): we only ever verified its first 32 B; its tail sits in
// flash right before 1.hex. Maps the corruption boundary on the known-good path.
void fs_scan_file(const char *filename) {
  pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
  if (fd == NULL) {
    printf("[FSSCAN] %s OPEN FAILED\r\n", filename);
    return;
  }
  const uint32_t CH = 2048;
  uint32_t size = fd->size;
  int first_bad = -1;
  uint32_t total_bad = 0;
  for (uint32_t off = 0; off < size; off += CH) {
    uint32_t n = (size - off) < CH ? (size - off) : CH;
    pi_fs_seek(fd, off);
    pi_fs_read(fd, buffer, n);
    const float *f = (const float *)buffer;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n / 4; i++) {
      float v = f[i];
      if (!(v > -100.0f && v < 100.0f)) { // catches nan/inf/huge
        bad++;
        if (first_bad < 0)
          first_bad = (int)(off + i * 4);
      }
    }
    total_bad += bad;
  }
  printf("[FSSCAN] %s size=%u first_bad_byte=%d total_bad_floats=%u\r\n", filename, (unsigned)size, first_bad,
         (unsigned)total_bad);
  pi_fs_close(fd);
}

// DEBUG: read the three readfs files in REVERSE order (weight, bias, input)
// using the FC-side pi_fs_read (the loader uses the CLUSTER pi_cl_fs_read). Runs
// in FC context (call from main after open_fs). Discriminates:
//   - 1.hex correct here but garbage in load  => cluster-read or read-ORDER bug.
//   - 1.hex garbage here too                  => data not in flash (flash/image bug).
void fs_order_test(void) {
  const char *names[3] = {"1.hex", "2.hex", "0.hex"};
  for (int f = 0; f < 3; f++) {
    pi_fs_file_t *fd = pi_fs_open(&fs, names[f], 0);
    if (fd == NULL) {
      printf("[FSORDER] %s: OPEN FAILED\r\n", names[f]);
      continue;
    }
    uint8_t local[32];
    pi_fs_seek(fd, 0);
    int32_t n = pi_fs_read(fd, local, 32);
    const float *lf = (const float *)local;
    printf("[FSORDER] %s (FC read, order#%d) size=%u n=%d:", names[f], f, (unsigned)fd->size, (int)n);
    for (int i = 0; i < 8; i++)
      printf(" %.5f", lf[i]);
    printf("\r\n");
    pi_fs_close(fd);
  }
}

size_t load_file_to_local(const void *dest, const char *filename) {
  pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
  if (fd == NULL) {
    printf("ERROR: Cannot open file %s! Exiting...", filename);
    pmsis_exit(-4);
  }

  const size_t size = fd->size;
  size_t remaining_size = size;
  size_t offset = 0;
  pi_cl_fs_req_t req;

  while (offset < size) {
    remaining_size = size - offset;
    size_t load_size =
        BUFFER_SIZE < remaining_size ? BUFFER_SIZE : remaining_size;
    pi_cl_fs_read(fd, buffer, load_size, &req);
    pi_cl_fs_wait(&req);
    memcpy(dest + offset, buffer, load_size);
    offset += load_size;
  }

  return offset;
}
