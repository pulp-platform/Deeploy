# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _DebugPrintTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        data_out._type = data_in._type

        operatorRepresentation['data_in_signed'] = data_in._signed
        operatorRepresentation['offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        operatorRepresentation['output_name'] = data_out.name

        return ctxt, operatorRepresentation, []


referenceTemplate = _DebugPrintTemplate("""
<%
tensor_type = "Input" if "input" in nodeName else "Output"
tensor_name = nodeName.replace("_input", "").replace("_output", "")
%>

// Debug Print (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_out} = ${data_in};
% if output_name != data_out:
    ${output_name} = ${data_out};
%endif
    deeploy_log("[DEBUG] ${tensor_type} ${tensor_name} (Buffer ${data_in}, Signed: ${data_in_signed}):\\r\\n");

    %if channels_first:
    %if data_in_signed:
        PrintMatrix_s${data_in_type.referencedType.typeWidth}_NCHW(${data_in}, ${batch}, ${dim_im_in_ch}, ${dim_im_in_x}, ${dim_im_in_y}, ${offset});
    %else:
        PrintMatrix_u${data_in_type.referencedType.typeWidth}_NCHW((uint${data_in_type.referencedType.typeWidth}_t *) ${data_in}, ${batch}, ${dim_im_in_ch}, ${dim_im_in_x}, ${dim_im_in_y}, ${offset});
    %endif
    %else:
    %if data_in_signed:
        PrintMatrix_s${data_in_type.referencedType.typeWidth}_NHWC(${data_in}, ${batch}, ${dim_im_in_ch}, ${dim_im_in_x}, ${dim_im_in_y}, ${offset});
    %else:
        PrintMatrix_u${data_in_type.referencedType.typeWidth}_NHWC((uint${data_in_type.referencedType.typeWidth}_t *) ${data_in}, ${batch}, ${dim_im_in_ch}, ${dim_im_in_x}, ${dim_im_in_y}, ${offset});
    %endif
    %endif
END_SINGLE_CORE
""")
