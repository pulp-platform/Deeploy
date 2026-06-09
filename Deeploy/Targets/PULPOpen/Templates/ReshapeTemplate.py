# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.Generic.Templates.ReshapeTemplate import _ReshapeTemplate as _GenericReshapeTemplate


class _ReshapeTemplate(_GenericReshapeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        ctxt, operatorRepresentation, _ = super().alignToContext(ctxt, operatorRepresentation)

        # Get buffers
        bufferIn = ctxt.lookup(operatorRepresentation['data_in'])
        assert isinstance(bufferIn, VariableBuffer)

        bufferOut = ctxt.lookup(operatorRepresentation['data_out'])
        assert isinstance(bufferOut, VariableBuffer)

        # HACK: Tiling wasn't updated in the Fix aliasing PR so we have to still
        #       set the _alias argument
        bufferOut._alias = bufferIn.name

        return ctxt, operatorRepresentation, []


referenceTemplate = _ReshapeTemplate("""
// Reshape (Name: ${nodeName}, Op: ${nodeOp})
${data_out} = ${data_in};
""")
