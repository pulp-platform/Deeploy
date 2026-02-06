# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.Generic.Templates.ReshapeTemplate import _ReshapeTemplate


class _SnitchReshapeTemplate(_ReshapeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        ctxt, operatorRepresentation, _ = super().alignToContext(ctxt, operatorRepresentation)

        # Calculate size for multi-core parallel copy
        bufferIn = ctxt.lookup(operatorRepresentation['data_in'])
        assert isinstance(bufferIn, VariableBuffer)
        operatorRepresentation['size'] = int(np.prod(bufferIn.shape))

        return ctxt, operatorRepresentation, []


# Reshape uses multi-core parallel copy
# When aliases work (internal nodes), this copies between same memory (no-op effect)
# When aliases don't work (global I/O), this copies data correctly
referenceTemplate = _SnitchReshapeTemplate("""
// Reshape (Name: ${nodeName}, Op: ${nodeOp})
{
    uint32_t core_id = snrt_cluster_core_idx();
    uint32_t num_cores = snrt_cluster_compute_core_num();
    uint32_t total = ${size};
    uint32_t chunk = total / num_cores;
    uint32_t start = core_id * chunk;
    uint32_t end = (core_id == num_cores - 1) ? total : start + chunk;
    for (uint32_t i = start; i < end; i++) {
        ${data_out}[i] = ${data_in}[i];
    }
}
""")
