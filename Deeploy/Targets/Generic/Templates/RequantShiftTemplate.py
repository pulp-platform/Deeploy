# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RequantShiftTemplate(NodeTemplate):

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
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * operatorRepresentation['n_levels'] // 2

        operatorRepresentation['output_min'] = -(operatorRepresentation['n_levels'] // 2)
        operatorRepresentation['output_max'] = (operatorRepresentation['n_levels'] // 2) - 1

        return ctxt, operatorRepresentation, []


referenceTemplate = _RequantShiftTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D
%>

// RequantShift (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    % if channels_first:
    RequantShift_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NCHW(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channel_width}, ${input_offset}, ${output_offset}, ${output_min}, ${output_max}, 1);
    % else:
    RequantShift_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NHWC(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channels}, ${input_offset}, ${output_offset}, ${output_min}, ${output_max}, 1);
    %endif
END_SINGLE_CORE
""")
