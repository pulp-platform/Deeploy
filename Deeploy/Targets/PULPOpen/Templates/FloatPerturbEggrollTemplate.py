# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatPerturbEggrollTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Add the node's unique ID to help create a unique seed_${nodeName}.
        operatorRepresentation['node_id'] = operatorRepresentation['nodeIdx']
        return ctxt, operatorRepresentation, []


# TODO: No loop unrolling optimization yet
referenceTemplate = _FloatPerturbEggrollTemplate("""
// Perturb Eggroll (Name: ${nodeName}, Op: ${nodeOp})
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk_a = (${sizeA} >> ${nodeName}_log2Core) + ((${sizeA} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start_a = (uint32_t) MIN(${nodeName}_chunk_a*${nodeName}_core_id, (uint32_t) ${sizeA});
uint32_t ${nodeName}_chunk_stop_a = (uint32_t) MIN(${nodeName}_chunk_start_a + ${nodeName}_chunk_a, (uint32_t) ${sizeA});
uint32_t ${nodeName}_local_size = ${nodeName}_chunk_stop_a - ${nodeName}_chunk_start_a;

uint32_t ${nodeName}_chunk_b = ( ${sizeB} >> ${nodeName}_log2Core) + ((${sizeB} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start_b = (uint32_t) MIN(${nodeName}_chunk_b*${nodeName}_core_id, (uint32_t) ${sizeB});
uint32_t ${nodeName}_chunk_stop_b = (uint32_t) MIN(${nodeName}_chunk_start_b + ${nodeName}_chunk_b, (uint32_t) ${sizeB});
uint32_t ${nodeName}_local_size_b = ${nodeName}_chunk_stop_b - ${nodeName}_chunk_start_b;
// pick large enough stride to minimize correlation between nodes.

float32_t *${nodeName}data_out;

uint32_t chunk_seed_a = seed + i*${nodeName}_chunk_start_a + (${node_id} * 104729);
uint32_t chunk_seed_b = seed + i*${nodeName}_chunk_start_b + (${node_id} * 104730);

GenEggrollPerturbation((float32_t *) & ${a_out}[${nodeName}_chunk_start_a],
                        chunk_seed_a,
                        ${nodeName}_local_size_a);
}
GenEggrollPerturbation((float32_t *) & ${b_out}[${nodeName}_chunk_start_b],
                        chunk_seed_b,
                        ${nodeName}_local_size_b);
}

""")