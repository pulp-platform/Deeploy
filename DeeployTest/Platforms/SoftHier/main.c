/*
 * ----------------------------------------------------------------------
 *
 * File: main.c
 *
 * Last edited: 26.05.2025
 *
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 *
 * Author: Bowen Wang (bowwang@iis.ee.ethz.ch), ETH Zurich
 *
 * ----------------------------------------------------------------------
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

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "flex_cluster_arch.h"

// Deeploy-generated
#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"


int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    flex_alloc_init();
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
    flex_intra_cluster_sync();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/
    uint32_t CID = flex_get_cluster_id();//Get cluster ID

    if(CID==0){ // only allow cluster 0 to work  
        if (flex_is_dm_core()) { // allow dm core to init network and dma
            printf("[main.c] >>> Initializing network...\n\n");
            InitNetwork(0, 1);

            printf("[main.c] >>> original data _in0: 0x%8x, _in1: 0x%8x\n", (uint32_t)testInputVector0, (uint32_t)testInputVector1);
            printf("[main.c] >>> allocated mem _in0: 0x%8x, _in1: 0x%8x\n\n", (uint32_t)DeeployNetwork_inputs[0], (uint32_t)DeeployNetwork_inputs[1]);

            for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
                // original data in HBM (placed by loader)
                void *ori_addr = testInputVector[buf];

                if ((uint64_t)DeeployNetwork_inputs[buf] < (uint64_t)ARCH_HBM_START_BASE){
                    // bowwang: Trigger DMA transaction: move from HBM to L1
                    uint64_t mask = 0x00000000ffffffff;
                    uint64_t masked_addr = (uint64_t)ori_addr & mask;
                    flex_dma_async_1d(      DeeployNetwork_inputs[buf],
                                            masked_addr, 
                                            DeeployNetwork_inputs_bytes[buf]);
                    //Wait all DMA transaction done
                    flex_dma_async_wait_all();
                } 
                else {
                    uint64_t *dst_addr = DeeployNetwork_inputs[buf];
                    // perform mem_copy with a single core
                    for (uint32_t i=0; i<(DeeployNetwork_inputs_bytes[buf]+7)/8; i++){
                        uint64_t data = ((uint64_t *)ori_addr)[i];
                        dst_addr[i] = data;
                    }
                }
            }
            
        }
        flex_intra_cluster_sync();//Cluster barrier

        if (flex_is_first_core()) { // allow core 0 to compute
            printf("[main.c] >>> Running network...\n\n");

            RunNetwork(0, 1);

        }
        flex_intra_cluster_sync();//Cluster barrier

        

        // verification
        int32_t tot_err = 0;
        uint32_t tot = 0;
        OUTPUTTYPE diff;
        OUTPUTTYPE expected, actual;

        if (flex_is_first_core()){
            for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
                tot += DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);
                for (uint32_t i = 0; i < DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
                    expected = ((OUTPUTTYPE *)testOutputVector[buf])[i];
                    actual = ((OUTPUTTYPE *)DeeployNetwork_outputs[buf])[i];
                    diff = expected - actual;
                    if (diff != 0){
                        tot_err += 1;
                        printf("Expected: %4d  ", expected);
                        printf("Actual: %4d  ", actual);
                        printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
                    }
                }
            }
            printf("Errors: %d out of %d \r\n", tot_err, tot);
        }
        flex_intra_cluster_sync();//Cluster barrier
    }
    
    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}