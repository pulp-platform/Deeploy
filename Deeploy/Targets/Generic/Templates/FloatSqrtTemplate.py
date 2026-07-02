# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.Generic.Templates.FloatUnaryTemplate import _FloatUnaryTemplate


class _SqrtTemplate(_FloatUnaryTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt, operatorRepresentation, deps = super().alignToContext(ctxt, operatorRepresentation)

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['data_type'] = data_in._type.typeName

        return ctxt, operatorRepresentation, deps


referenceTemplate = _SqrtTemplate("""
// Sqrt (Name: ${nodeName}, Op: ${nodeOp})
Sqrt_fp${type_width}_fp${type_width}(${data_in}, ${data_out}, ${size});
""")
