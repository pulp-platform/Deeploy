# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.Generic.Templates.ReshapeTemplate import _ReshapeTemplate


class _SnitchReshapeTemplate(_ReshapeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        ctxt, operatorRepresentation, _ = super().alignToContext(ctxt, operatorRepresentation)

        bufferIn = ctxt.lookup(operatorRepresentation['data_in'])
        assert isinstance(bufferIn, VariableBuffer)
        bufferOut = ctxt.lookup(operatorRepresentation['data_out'])
        assert isinstance(bufferOut, VariableBuffer)

        # Set alias so input and output share the same memory
        bufferOut._alias = bufferIn.name

        return ctxt, operatorRepresentation, []


# Reshape only reinterprets tensor shape without modifying data.
# Uses SkipTransformer (no DMA), consistent with PULPOpen.
referenceTemplate = _SnitchReshapeTemplate("""
// Reshape (Name: ${nodeName}, Op: ${nodeOp})
${data_out} = ${data_in};
""")
