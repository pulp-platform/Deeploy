# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _UniformRequantShiftTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation["signedI"] = data_in._type.referencedType.typeMin < 0
        operatorRepresentation["signedO"] = data_out._type.referencedType.typeMin < 0

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * operatorRepresentation['n_levels'] // 2

        if operatorRepresentation["signed"]:
            operatorRepresentation['output_min'] = -(operatorRepresentation['n_levels'] // 2)
            operatorRepresentation['output_max'] = (operatorRepresentation['n_levels'] // 2) - 1
        else:
            operatorRepresentation['output_min'] = 0
            operatorRepresentation['output_max'] = operatorRepresentation['n_levels'] - 1

        operatorRepresentation['mul_immediate'] = ctxt.lookup(operatorRepresentation['mul']).values.flatten()[0]
        operatorRepresentation['add_immediate'] = ctxt.lookup(operatorRepresentation['add']).values.flatten()[0]

        # JUNGVI: Don't tile the mul and add tensors in case of uniform requantization
        mul_buffer = ctxt.lookup(operatorRepresentation['mul'])
        add_buffer = ctxt.lookup(operatorRepresentation['add'])
        mul_buffer._deploy = False
        add_buffer._deploy = False

        return ctxt, operatorRepresentation, []


referenceTemplate = _UniformRequantShiftTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D

inSignage = "s" if signedI else "u"
outSignage = "s" if signedO else "u"
mul_int_immediate = int(mul_immediate)
add_int_immediate = int(add_immediate)
%>

// UniformRequantShift (Name: ${nodeName}, Op: ${nodeOp})
UniformRequantShift_${inSignage}${data_in_type.referencedType.typeWidth}_${outSignage}${data_out_type.referencedType.typeWidth}(${data_in}, ${size}, ${mul_int_immediate}, ${add_int_immediate}, ${data_out}, ${log2Dstring}, ${channel_width}, 0, 0 , ${output_min}, ${output_max}, 1);
""")
