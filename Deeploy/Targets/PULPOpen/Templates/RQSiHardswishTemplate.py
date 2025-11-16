# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RQSiHardswishTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = (data_out._signed == 0) * int(data_out.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _RQSiHardswishTemplate("""
// RequantizediHardswish (Name: ${nodeName}, Op: ${nodeOp})
RQiHardswish_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_plp(${data_in}, ${data_out}, ${size}, ${one_over_six}, ${three},${six}, ${int(mul)}, ${int(add)}, ${int(shift)});
""")
