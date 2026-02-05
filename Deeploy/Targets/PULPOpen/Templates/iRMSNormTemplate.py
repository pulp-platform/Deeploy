# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _iRMSNormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        return ctxt, operatorRepresentation, []


referenceTemplate = _iRMSNormTemplate("""
// iRMSnorm (Name: ${nodeName}, Op: ${nodeOp})
iRMSnorm_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_plp(${data_in}, ${data_out}, ${weight}, ${size}, ${lastDimLength}, ${log2D});
""")
