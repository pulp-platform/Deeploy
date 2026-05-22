# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _GlobalMaxPoolTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> tuple[NetworkContext, dict, list[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['type_width'] = data_in._type.referencedType.typeWidth
        return ctxt, operatorRepresentation, []


referenceTemplate = _GlobalMaxPoolTemplate("""
// Global Max Pool 1D (Name: ${nodeName}, Op: ${nodeOp})
GlobalMaxPool_fp${type_width}_fp${type_width}(
    ${data_in}, ${data_out}, ${batch_size}, ${num_channels}, ${spatial_size});
""")