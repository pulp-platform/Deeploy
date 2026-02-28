# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatPerturbNormalTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Add the node's unique ID to help create a unique seed_${nodeName}.
        operatorRepresentation['node_id'] = operatorRepresentation['nodeIdx']
        return ctxt, operatorRepresentation, []


# TODO: No loop unrolling optimization yet
referenceTemplate = _FloatPerturbNormalTemplate("""
// PerturbNormal (Name: ${nodeName}, Op: ${nodeOp})
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});
uint32_t ${nodeName}_local_size = ${nodeName}_chunk_stop - ${nodeName}_chunk_start;

uint32_t chunk_seed = seed + (${nodeName}_chunk_start * ${node_id}) + (${node_id} * 104729);

// pick large enough stride to minimize correlation between nodes.
ApplyGaussianPerturbation(
    (const float32_t *) &${data_in}[${nodeName}_chunk_start],
    (float32_t *) &${data_out}[${nodeName}_chunk_start],
    chunk_seed,
    perturbation_sign, // globally defined in DeedeployTest main
    ${nodeName}_local_size,
    ${eps}f,
);
"""
)