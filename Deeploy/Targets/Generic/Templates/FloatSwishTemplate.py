# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _SigmoidTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> tuple[NetworkContext, dict, list[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['size'] = int(np.prod(data_in.shape))
        operatorRepresentation['type_width'] = data_in._type.referencedType.typeWidth
        return ctxt, operatorRepresentation, []


referenceTemplate = _SigmoidTemplate("""
// Swish (Name: ${nodeName}, Op: ${nodeOp})
Swish_fp${type_width}_fp${type_width}(${data_in}, ${data_out}, ${alpha}, ${size});
""")
