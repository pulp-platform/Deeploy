# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _AveragePoolTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> tuple[NetworkContext, dict, list[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['type_width'] = data_in._type.referencedType.typeWidth
        return ctxt, operatorRepresentation, []


referenceTemplate1d = _AveragePoolTemplate("""
// Average Pool 1D (Name: ${nodeName}, Op: ${nodeOp})
AveragePool1d_fp${type_width}_fp${type_width}(
    ${data_in}, ${data_out}, ${batch_size}, ${num_channels}, ${length}, ${kernel_shape[0]},
    ${strides[0]}, ${pads[0]}, ${pads[1]});
""")

referenceTemplate2d = _AveragePoolTemplate("""
// Average Pool 2D (Name: ${nodeName}, Op: ${nodeOp})
AveragePool2d_fp${type_width}_fp${type_width}(
    ${data_in}, ${data_out}, ${batch_size}, ${num_channels}, ${height}, ${width},
    ${kernel_shape[0]}, ${kernel_shape[1]}, ${strides[0]}, ${strides[1]},
    ${pads[0]}, ${pads[1]}, ${pads[2]}, ${pads[3]});
""")
