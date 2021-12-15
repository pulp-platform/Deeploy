/* =====================================================================
 * Title:        M4HSA_s8.c
 * Description:
 *
 * Date:         08.02.2023
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

#include "DeeployMath.h"

dump(timer_cycle, 0);
dump(timer_instr, 1);

void M1HSA_s8_ITA(int8_t const *__restrict__ pSrcQ,
                  int8_t const *__restrict__ pSrcK, int8_t *__restrict__ pBuf,
                  uint32_t S, uint32_t E, uint32_t P,
                  ita_quant_t const *__restrict__ quant_param,
                  int8_t *__restrict__ pDst, int8_t Q_offset, int8_t K_offset,
                  int8_t output_offset, uint32_t core_id,
                  __attribute__((unused)) uint32_t numThreads) {

  ita_data_t ita_data;
  uint32_t i = 0;
  uint32_t j = 0;

  ITA_getStruct(&ita_data, pBuf, S, E, P);
  // if (core_id == 0) ITA_printAddresses(&ita_data);

  // Copy the keys (Q) and queries/values (K) data to the L2 buffer
  if (core_id == 0)
    ITA_copyInput(ita_data.q, pSrcQ, S, E, Q_offset);
#if USE_DMA
  if (core_id == 0) {
#else
  if ((core_id == 1) | (numThreads == 1)) {
#endif
    ITA_copyInput(ita_data.k, pSrcK, S, E, K_offset);
  }

  mempool_barrier(numThreads);

  mempool_stop_benchmark();
  mempool_start_benchmark();

  mempool_timer_t instr_init = 0, instr_end = 0;
  mempool_timer_t timer_init = 0, timer_end = 0;

  if (core_id == 0) {
    // Configure ITA core 0
    ITA_SetShape(ITA0, S, E, P);
    ITA_SetStartAddress(ITA0, (uint32_t)pBuf);
    ITA_SetOutAddress(ITA0, (uint32_t)pDst);
    ITA_SetRQSAddress(ITA0, (uint32_t)quant_param);
    ITA_SetIter(ITA0, 1);

    // Run one iteration
    instr_init = read_csr(minstret);
    timer_init = read_csr(mcycle);

    ITA_Start(ITA0);

    while (!ITA_IsDone(ITA0)) {
      mempool_wait(16);
    }
  }

  mempool_barrier(numThreads);

  mempool_stop_benchmark();
  mempool_start_benchmark();
  if (core_id == 0) {
    timer_end = read_csr(mcycle);
    instr_end = read_csr(minstret);
    dump_timer_cycle(timer_end - timer_init);
    dump_timer_instr(instr_end - instr_init - 2);

    // Add offset to output matrix
    if (output_offset != 0) {
      for (i = 0; i < S; ++i) {
        for (j = 0; j < E; ++j) {
          pDst[i * E + j] += output_offset;
        }
      }
    }
  }
}

void M2HSA_s8_ITA(int8_t const *__restrict__ pSrcQ,
                  int8_t const *__restrict__ pSrcK, int8_t **__restrict__ pBuf,
                  uint32_t S, uint32_t E, uint32_t P,
                  ita_quant_t const **__restrict__ quant_params,
                  int8_t *__restrict__ pDst, int8_t Q_offset, int8_t K_offset,
                  int8_t output_offset, uint32_t core_id,
                  __attribute__((unused)) uint32_t numThreads) {

  ita_data_t ita_data;
  uint32_t i = 0;
  uint32_t j = 0;
  ITA_TypeDef *ita_inst[] = {ITA0, ITA1};
  uint8_t ita_h = sizeof(ita_inst) / sizeof(ITA_TypeDef *);

  ITA_getStruct(&ita_data, pBuf[0], S, E, P);
  // if (core_id == 0) ITA_printAddresses(&ita_data);

  // Copy the keys (Q) and queries/values (K) data to the L2 buffer
  if (core_id == 0)
    ITA_copyInput(ita_data.q, pSrcQ, S, E, Q_offset);
#if USE_DMA
  if (core_id == 0) {
#else
  if ((core_id == 1) | (numThreads == 1)) {
#endif
    ITA_copyInput(ita_data.k, pSrcK, S, E, K_offset);
  }

  // WIESP: All ITA cores fetch the Q and K vector always from the address
  // specified to core 0, hence we must make sure that this is valid
  if (core_id == 0)
    ITA_SetStartAddress(ita_inst[core_id], (uint32_t)pBuf[core_id]);

  mempool_barrier(numThreads);

  mempool_stop_benchmark();
  mempool_start_benchmark();

  mempool_timer_t instr_init = 0, instr_end = 0;
  mempool_timer_t timer_init = 0, timer_end = 0;

  if (core_id < ita_h) {
    // Configure ITA cores
    ITA_SetShape(ita_inst[core_id], S, E, P);
    ITA_SetStartAddress(ita_inst[core_id], (uint32_t)pBuf[core_id]);
    ITA_SetOutAddress(ita_inst[core_id], (uint32_t)(pDst + core_id * S * E));
    ITA_SetRQSAddress(ita_inst[core_id], (uint32_t)quant_params[core_id]);
    ITA_SetIter(ita_inst[core_id], 1);

    // Run one iteration
    instr_init = read_csr(minstret);
    timer_init = read_csr(mcycle);

    ITA_Start(ita_inst[core_id]);

    while (!ITA_IsDone(ita_inst[core_id])) {
      mempool_wait(16);
    }
  }
  mempool_barrier(numThreads);

  mempool_stop_benchmark();
  mempool_start_benchmark();
  if (core_id < ita_h) {
    timer_end = read_csr(mcycle);
    instr_end = read_csr(minstret);
    dump_timer_cycle(timer_end - timer_init);
    dump_timer_instr(instr_end - instr_init - 2);

    // Add offset to output matrix
    if (output_offset != 0) {
      for (i = 0; i < S; ++i) {
        for (j = 0; j < E; ++j) {
          pDst[core_id * S * E + i * E + j] += output_offset;
        }
      }
    }
  }
}

void M4HSA_s8_ITA(int8_t const *__restrict__ pSrcQ,
                  int8_t const *__restrict__ pSrcK, int8_t **__restrict__ pBuf,
                  uint32_t S, uint32_t E, uint32_t P,
                  ita_quant_t const **__restrict__ quant_params,
                  int8_t *__restrict__ pDst, int8_t Q_offset, int8_t K_offset,
                  int8_t output_offset, uint32_t core_id,
                  __attribute__((unused)) uint32_t numThreads) {

  ita_data_t ita_data;
  uint32_t i = 0;
  uint32_t j = 0;
  ITA_TypeDef *ita_inst[] = {ITA0, ITA1, ITA2, ITA3};
  uint8_t ita_h = sizeof(ita_inst) / sizeof(ITA_TypeDef *);

  ITA_getStruct(&ita_data, pBuf[0], S, E, P);
  // if (core_id == 0) ITA_printAddresses(&ita_data);

  // Copy the keys (Q) and queries/values (K) data to the L2 buffer
  if (core_id == 0)
    ITA_copyInput(ita_data.q, pSrcQ, S, E, Q_offset);
#if USE_DMA
  if (core_id == 0) {
#else
  if ((core_id == 1) | (numThreads == 1)) {
#endif
    ITA_copyInput(ita_data.k, pSrcK, S, E, K_offset);
  }

  // WIESP: All ITA cores fetch the Q and K vector always from the address
  // specified to core 0, hence we must make sure that this is valid
  if (core_id == 0)
    ITA_SetStartAddress(ita_inst[core_id], (uint32_t)pBuf[core_id]);

  mempool_barrier(numThreads);

  mempool_stop_benchmark();
  mempool_start_benchmark();

  mempool_timer_t instr_init = 0, instr_end = 0;
  mempool_timer_t timer_init = 0, timer_end = 0;

  if (core_id < ita_h) {
    // Configure ITA cores
    ITA_SetShape(ita_inst[core_id], S, E, P);
    ITA_SetStartAddress(ita_inst[core_id], (uint32_t)pBuf[core_id]);
    ITA_SetOutAddress(ita_inst[core_id], (uint32_t)(pDst + core_id * S * E));
    ITA_SetRQSAddress(ita_inst[core_id], (uint32_t)quant_params[core_id]);
    ITA_SetIter(ita_inst[core_id], 1);

    // Run one iteration
    instr_init = read_csr(minstret);
    timer_init = read_csr(mcycle);

    ITA_Start(ita_inst[core_id]);

    while (!ITA_IsDone(ita_inst[core_id])) {
      mempool_wait(16);
    }
  }
  mempool_barrier(numThreads);

  mempool_stop_benchmark();
  mempool_start_benchmark();
  if (core_id < ita_h) {
    timer_end = read_csr(mcycle);
    instr_end = read_csr(minstret);
    dump_timer_cycle(timer_end - timer_init);
    dump_timer_instr(instr_end - instr_init - 2);

    // Add offset to output matrix
    if (output_offset != 0) {
      for (i = 0; i < S; ++i) {
        for (j = 0; j < E; ++j) {
          pDst[core_id * S * E + i * E + j] += output_offset;
        }
      }
    }
  }
}
