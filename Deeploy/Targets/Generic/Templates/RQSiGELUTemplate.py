# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RQSiGELUTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _RQSiGELUTemplate("""
// RequantizediGELU (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE RQGELU_s${data_in_type.referencedType.typeWidth}_s${data_in_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${size}, ${b}, ${one}, ${input_offset}, ${output_offset}, ${mul}, ${add}, ${shift});
""")
