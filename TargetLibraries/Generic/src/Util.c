/* =====================================================================
 * Title:        Util.c
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

#include "DeeployBasicMath.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// WIESEP: Provide implementation for the generic platform as it has no
// dedicated library
#ifdef DEEPLOY_GENERIC_PLATFORM
int deeploy_log(const char *__restrict fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vprintf(fmt, args);
  va_end(args);
  return ret;
}
void *deeploy_malloc(const size_t size) { return malloc(size); }
void deeploy_free(void *const ptr) { free(ptr); }
#else
extern int deeploy_log(const char *__restrict fmt, ...);
extern void *deeploy_malloc(const size_t size);
extern void deeploy_free(void *const ptr);
#endif

void PrintMatrix_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t N,
                         uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log(
              "%4d ",
              (int8_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] + offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_s8_NHWC(int8_t const *__restrict__ pSrcA, uint32_t N,
                         uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log(
              "%4d ",
              (int8_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] + offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_s16_NCHW(int16_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log(
              "%6hd ",
              (int16_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] + offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_s16_NHWC(int16_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log(
              "%6hd ",
              (int16_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] + offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_s32_NCHW(int32_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log(
              "%11" PRId32 " ",
              (int32_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] + offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_s32_NHWC(int32_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log(
              "%11" PRId32 " ",
              (int32_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] + offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintArray_s8(int8_t const *__restrict__ pSrcA, uint32_t N,
                   int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    deeploy_log("%4d ", (int8_t)(pSrcA[n] + offset));
  }
  deeploy_log("\r\n");
}

void PrintArray_s16(int16_t const *__restrict__ pSrcA, uint32_t N,
                    int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    deeploy_log("%6hd ", (int16_t)(pSrcA[n] + offset));
  }
  deeploy_log("\r\n");
}

void PrintArray_s32(int32_t const *__restrict__ pSrcA, uint32_t N,
                    int32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    deeploy_log("%11" PRId32 " ", (int32_t)(pSrcA[n] + offset));
  }
  deeploy_log("\r\n");
}

void PrintMatrix_u8_NCHW(uint8_t const *__restrict__ pSrcA, uint32_t N,
                         uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log(
              "%4u ",
              (uint8_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] + offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_u8_NHWC(uint8_t const *__restrict__ pSrcA, uint32_t N,
                         uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log(
              "%4u ",
              (uint8_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] + offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_u16_NCHW(uint16_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log("%6hu ",
                      (uint16_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] +
                                 offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_u16_NHWC(uint16_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log("%6hu ",
                      (uint16_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] +
                                 offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_u32_NCHW(uint32_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log("%11" PRIu32 " ",
                      (uint32_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] +
                                 offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintMatrix_u32_NHWC(uint32_t const *__restrict__ pSrcA, uint32_t N,
                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    if (N > 0)
      deeploy_log("[\r\n");

    for (uint32_t c = 0; c < C; c++) {
      if (N > 0) {
        deeploy_log("  [\r\n  ");
      } else if (C > 0) {
        deeploy_log("[\r\n");
      }
      for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
          deeploy_log("%11" PRIu32 " ",
                      (uint32_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] +
                                 offset));
        }

        if (N > 0) {
          deeploy_log("\r\n  ");
        } else {
          deeploy_log("\r\n");
        }
      }
      if (C > 0)
        deeploy_log("]\r\n");
    }

    if (N > 0)
      deeploy_log("]\r\n");
  }
}

void PrintArray_u8(uint8_t const *__restrict__ pSrcA, uint32_t N,
                   uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    deeploy_log("%4u ", (uint8_t)(pSrcA[n] + offset));
  }
  deeploy_log("\r\n");
}

void PrintArray_u16(uint16_t const *__restrict__ pSrcA, uint32_t N,
                    uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    deeploy_log("%6hu ", (uint16_t)(pSrcA[n] + offset));
  }
  deeploy_log("\r\n");
}

void PrintArray_u32(uint32_t const *__restrict__ pSrcA, uint32_t N,
                    uint32_t offset) {
  for (uint32_t n = 0; n < N; n++) {
    deeploy_log("%11" PRIu32 " ", (uint32_t)(pSrcA[n] + offset));
  }
  deeploy_log("\r\n");
}
