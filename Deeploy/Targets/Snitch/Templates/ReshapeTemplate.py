# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer


class _SnitchReshapeTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # SCHEREMO: Selectively mark 'indices' dead, since we don't need them
        if 'indices' in operatorRepresentation.keys():
            ctxt.globalObjects[operatorRepresentation['indices']]._deploy = False
            ctxt.globalObjects[operatorRepresentation['indices']]._live = False

        # Same for "shape"
        if "shape" in operatorRepresentation.keys():
            ctxt.globalObjects[operatorRepresentation["shape"]]._deploy = False
            ctxt.globalObjects[operatorRepresentation["shape"]]._live = False

        bufferIn = ctxt.lookup(operatorRepresentation['data_in'])
        assert isinstance(bufferIn, VariableBuffer)
        bufferOut = ctxt.lookup(operatorRepresentation['data_out'])
        assert isinstance(bufferOut, VariableBuffer)

        # Link aliases to each buffer
        bufferIn.aliases.add(bufferOut.name)
        bufferOut.aliases.add(bufferIn.name)

        return ctxt, operatorRepresentation, []


# Use snrt_cluster_core_idx() == 0 instead of SINGLE_CORE macro to avoid core_id dependency
referenceTemplate = _SnitchReshapeTemplate("""
// Reshape (Name: ${nodeName}, Op: ${nodeOp})
if (snrt_cluster_core_idx() == 0) { ${data_out} = ${data_in}; }
""")
