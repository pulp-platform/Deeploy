# ----------------------------------------------------------------------
#
# File: DebugPrintTemplate.py
#
# Last edited: 14.12.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

        operatorRepresentation['output_name'] = ctxt._mangle("outputs") + "[0]" if ctxt.outputs(
        )[0].name == data_out.name else ctxt._mangle(data_out.name)

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
