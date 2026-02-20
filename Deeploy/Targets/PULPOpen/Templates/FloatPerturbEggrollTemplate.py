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
);
"""
)