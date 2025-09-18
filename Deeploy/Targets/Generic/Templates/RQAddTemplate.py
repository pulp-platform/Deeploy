# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class RQAddTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedI2 = ctxt.lookup(operatorRepresentation['data_in_2'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['data_in_1'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['input_2_signed'] = signedI2
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []
