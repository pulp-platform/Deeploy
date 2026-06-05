# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import FloatImmediate
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

_REDUCTION_MAP = {
    "none": "SCATTER_REDUCTION_NONE",
    "add": "SCATTER_REDUCTION_ADD",
    "mul": "SCATTER_REDUCTION_MUL",
    "min": "SCATTER_REDUCTION_MIN",
    "max": "SCATTER_REDUCTION_MAX",
}


def _typeSuffix(ref_type) -> str:
    if issubclass(ref_type, FloatImmediate):
        return f'fp{ref_type.typeWidth}'
    elif ref_type.signed:
        return f's{ref_type.typeWidth}'
    else:
        return f'u{ref_type.typeWidth}'


class _ScatterTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['type_suffix'] = _typeSuffix(data_in._type.referencedType)
        operatorRepresentation['reduction_c'] = _REDUCTION_MAP[operatorRepresentation['reduction']]

        return ctxt, operatorRepresentation, []


referenceTemplate = _ScatterTemplate("""
// Scatter (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
Scatter_${type_suffix}(${data_in}, ${indices}, ${updates}, ${data_out}, ${ndim},
    (int32_t[]){${', '.join(str(s) for s in data_shape)}},
    (int32_t[]){${', '.join(str(s) for s in indices_shape)}},
    ${axis}, ${reduction_c}
);
END_SINGLE_CORE
""")
