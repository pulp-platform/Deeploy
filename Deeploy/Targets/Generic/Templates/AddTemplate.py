# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.CommonExtensions.NodeTemplate import ElementwiseTemplate
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation


class _AddTemplate(ElementwiseTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in_1 = ctxt.lookup(operatorRepresentation['data_in_1'])
        data_in_2 = ctxt.lookup(operatorRepresentation['data_in_2'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        input_1_offset = 0
        if hasattr(data_in_1, "_signed") and hasattr(data_in_1, "nLevels"):
            input_1_offset = (data_in_1._signed == 0) * int(data_in_1.nLevels / 2)
        input_2_offset = 0
        if hasattr(data_in_2, "_signed") and hasattr(data_in_2, "nLevels"):
            input_2_offset = (data_in_2._signed == 0) * int(data_in_2.nLevels / 2)
        output_offset = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            output_offset = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        operatorRepresentation['offset'] = input_1_offset + input_2_offset + output_offset

        return ctxt, operatorRepresentation, []


referenceTemplate = _AddTemplate("""
// Add (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i=0;i<${size};i++){
        ${data_out}[i] = ${data_in_1}[i] + ${data_in_2}[i] + ${offset};
    }
END_SINGLE_CORE
""")
