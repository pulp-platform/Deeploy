/* =====================================================================
 * Title:        ITA.h
 * Description:
 *
 * Date:         03.03.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
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

#ifndef __DEEPLOY_MATH_ITA_HEADER_
#define __DEEPLOY_MATH_ITA_HEADER_

/* Includes ------------------------------------------------------------------*/

#include "DeeployMath.h"

/* Exported constants --------------------------------------------------------*/

#define ITA0_BASE (0x40000040)
#define ITA1_BASE (0x40000070)
#define ITA2_BASE (0x400000A0)
#define ITA3_BASE (0x400000D0)

#define ITA0_L1_BASE (0xC0000)
#define ITA1_L1_BASE (0xD0000)
#define ITA2_L1_BASE (0xE0000)
#define ITA3_L1_BASE (0xF0000)

// ITA has 0x1000 bytes per matrix and OUT is matrix 0
#define ITA0_OUT (ITA0_L1_BASE + 0x3000)
#define ITA1_OUT (ITA1_L1_BASE + 0x3000)
#define ITA2_OUT (ITA2_L1_BASE + 0x3000)
#define ITA3_OUT (ITA3_L1_BASE + 0x3000)

// Default ITA configuration
#ifndef ITA_PE
#define ITA_PE 16 // Number of processing engines per ITA core
#endif

/* Exported macros -----------------------------------------------------------*/
#define MUL(X, Y) ((X) * (Y))

// clang-format off
#define SET_BIT(REG, BIT)     ((REG) |= (BIT))
#define CLEAR_BIT(REG, BIT)   ((REG) &= ~(BIT))
#define READ_BIT(REG, BIT)    ((REG) & (BIT))
#define CLEAR_REG(REG)        ((REG) = (0x0))
#define WRITE_REG(REG, VAL)   ((REG) = (VAL))
#define READ_REG(REG)         ((REG))

#define MODIFY_REG(REG, CLEARMASK, SETMASK) WRITE_REG((REG), (((READ_REG(REG)) & (~(CLEARMASK))) | (SETMASK)))
// clang-format on

/* Exported types ------------------------------------------------------------*/

typedef struct {
  int8_t *wo_weight;
  int8_t *wv_weight;
  int8_t *wk_weight;
  int8_t *q;
  int8_t *k;
  int8_t *wq_weight;
  int32_t *wo_bias;
  int32_t *wv_bias;
  int32_t *wk_bias;
  int32_t *wq_bias;
} ita_data_t;

typedef struct {
  uint8_t *eps_mult;
  uint8_t *right_shift;
  int32_t *add;
} ita_quant_t;

/* Exported variables --------------------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
void ITA_getStruct(ita_data_t *ita_data, int8_t *base_address, uint32_t S,
                   uint32_t E, uint32_t P);

void ITA_copyInput(int8_t *pDst, int8_t const *__restrict__ pSrc, uint32_t S,
                   uint32_t E, int8_t offset);

void ITA_printAddresses(ita_data_t *ita_data);

/* Peripheral Definition
 * ------------------------------------------------------*/
// Inspired by CMSIS Device Peripheral Access Layer Header Files.

// clang-format off
typedef struct
{
  volatile uint32_t STATE;          /*!< ITA STATE Register,                                        Address offset: 0x00 */
  volatile uint32_t START_ADDR;     /*!< ITA Start Address,                                         Address offset: 0x04 */
  volatile uint32_t OUT_ADDR;       /*!< ITA Out Address,                                           Address offset: 0x08 */
  volatile uint32_t RQS_ADDR;       /*!< ITA Requantization Parameter Address,                      Address offset: 0x0C */
  volatile uint32_t S;              /*!< ITA Sequence Length Register                               Address offset: 0x10 */
  volatile uint32_t E;              /*!< ITA Embedding Length Register                              Address offset: 0x14 */
  volatile uint32_t P;              /*!< ITA Projection Length Register                             Address offset: 0x18 */
} ITA_TypeDef;

#define ITA_CONFIG_START_Pos          (0U)
#define ITA_CONFIG_START_Msk          (0x1UL << ITA_CONFIG_START_Pos)           /*!< 0x00000001 */
#define ITA_CONFIG_START              ITA_CONFIG_START_Msk                     /*!< ITA Start Computation Flag */

#define ITA_CONFIG_BUSY_Pos           (1U)
#define ITA_CONFIG_BUSY_Msk           (0x1UL << ITA_CONFIG_BUSY_Pos)            /*!< 0x00000002 */
#define ITA_CONFIG_BUSY               ITA_CONFIG_BUSY_Msk                      /*!< ITA Busy Flag */

#define ITA_CONFIG_DONE_Pos           (2U)
#define ITA_CONFIG_DONE_Msk           (0x1UL << ITA_CONFIG_DONE_Pos)            /*!< 0x00000004 */
#define ITA_CONFIG_DONE               ITA_CONFIG_DONE_Msk                      /*!< ITA1 Done Flag */

#define ITA_CONFIG_SKIPSOFTMAX_Pos    (4U)
#define ITA_CONFIG_SKIPSOFTMAX_Msk    (0x1UL << ITA_CONFIG_SKIPSOFTMAX_Pos)     /*!< 0x00000010 */
#define ITA_CONFIG_SKIPSOFTMAX        ITA_CONFIG_SKIPSOFTMAX_Msk                /*!< ITA1 Skip Softmax Flag (unused) */

#define ITA_CONFIG_ITER_Pos           (5U)
#define ITA_CONFIG_ITER_Msk           (0x7UL << ITA_CONFIG_ITER_Pos)            /*!< 0x000000e0 */
#define ITA_CONFIG_ITER               ITA_CONFIG_ITER_Msk                       /*!< ITA Iteration [2:0] Bits (Value 0 -> 1 Iteration) */

#define ITA0              ((ITA_TypeDef *) ITA0_BASE)
#define ITA1              ((ITA_TypeDef *) ITA1_BASE)
#define ITA2              ((ITA_TypeDef *) ITA2_BASE)
#define ITA3              ((ITA_TypeDef *) ITA3_BASE)


static inline void ITA_Start(ITA_TypeDef *ITAx)
{
  // An explicit softmax state is no longer used, hence we always skip it.
  SET_BIT(ITAx->STATE, ITA_CONFIG_SKIPSOFTMAX);

  // Start ITA
  SET_BIT(ITAx->STATE, ITA_CONFIG_START);
}

static inline void ITA_SetStartAddress(ITA_TypeDef *ITAx, const uint32_t StartAddress)
{
  ITAx->START_ADDR = StartAddress;
}

static inline uint32_t ITA_GetStartAddress(const ITA_TypeDef *ITAx)
{
  return ITAx->START_ADDR;
}

static inline void ITA_SetOutAddress(ITA_TypeDef *ITAx, const uint32_t OutAddress)
{
  ITAx->OUT_ADDR = OutAddress;
}

static inline uint32_t ITA_GetOutAddress(const ITA_TypeDef *ITAx)
{
  return ITAx->OUT_ADDR;
}

static inline void ITA_SetRQSAddress(ITA_TypeDef *ITAx, const uint32_t RQSAddress)
{
  ITAx->RQS_ADDR = RQSAddress;
}

static inline uint32_t ITA_GetRQSAddress(const ITA_TypeDef *ITAx)
{
  return ITAx->RQS_ADDR;
}

static inline void ITA_SetShape(ITA_TypeDef *ITAx, uint32_t S, uint32_t E, uint32_t P)
{
  ITAx->S = S;
  ITAx->E = E;
  ITAx->P = P;
}

static inline uint32_t ITA_GetShape_S(const ITA_TypeDef *ITAx)
{
  return ITAx->S;
}

static inline uint32_t ITA_GetShape_E(const ITA_TypeDef *ITAx)
{
  return ITAx->E;
}

static inline uint32_t ITA_GetShape_P(const ITA_TypeDef *ITAx)
{
  return ITAx->P;
}

static inline uint32_t ITA_IsBusy(const ITA_TypeDef *ITAx)
{
  return (READ_BIT(ITAx->STATE, ITA_CONFIG_BUSY) == (ITA_CONFIG_BUSY));
}

static inline uint32_t ITA_IsDone(const ITA_TypeDef *ITAx)
{
  return (READ_BIT(ITAx->STATE, ITA_CONFIG_DONE) == (ITA_CONFIG_DONE));
}

static inline void ITA_SetIter(ITA_TypeDef *ITAx, uint32_t Counter)
{
  if (Counter > 0) {
    MODIFY_REG(ITAx->STATE, ITA_CONFIG_ITER, Counter - 1);
  }
}
// clang-format on

#endif //__DEEPLOY_MATH_ITA_HEADER_
