# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import FloatImmediate
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


def _typeSuffix(ref_type) -> str:
    if issubclass(ref_type, FloatImmediate):
        return f'fp{ref_type.typeWidth}'
    elif ref_type.signed:
        return f's{ref_type.typeWidth}'
    else:
        return f'u{ref_type.typeWidth}'


class _Col2ImTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['type_suffix'] = _typeSuffix(data_in._type.referencedType)

        return ctxt, operatorRepresentation, []


referenceTemplate = _Col2ImTemplate("""
// Col2Im (Name: ${nodeName}, Op: ${nodeOp})
Col2Im_${type_suffix}(
    ${data_in}, ${data_out},
    ${batch_size}, ${channels}, ${spatial_dims},
    (int32_t[]){${', '.join(str(s) for s in image_shape)}},
    (int32_t[]){${', '.join(str(s) for s in block_shape)}},
    (int32_t[]){${', '.join(str(s) for s in dilations)}},
    (int32_t[]){${', '.join(str(s) for s in pads)}},
    (int32_t[]){${', '.join(str(s) for s in strides)}}
);
""")
