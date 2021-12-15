# ----------------------------------------------------------------------
#
# File: ReduceMeanTemplate.py
#
# Last edited: 05.06.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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


class _ReduceMeanTemplate(NodeTemplate):

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
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _ReduceMeanTemplate("""
// ReduceMean (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
int32_t ${data_out}_accumulator = 0;
<%

reduceLength = 1
for i, axis in enumerate(axes):
    if axis < 0:
        axes[i] += len(data_in_shape)
    reduceLength = reduceLength * data_in_shape[axis]
%>
<%
    shapeStr = ''
    accessStr = ''
%>
% for idx, i in enumerate(data_in_shape[1:]):
<%
    shapeStr += '['+str(i)+']'
%>
% endfor
% for j in range(len(data_in_shape)):
<%
    accessStr += '[i_'+str(j)+']'
%>
% endfor
${data_out_type.typeName} dummy_${data_out} = ${data_out};

<%
restDims = set(list(range(len(data_in_shape)))).difference(set(axes))
%>
% for i in list(restDims):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator = ${input_offset}*${reduceLength};
% for i in list(axes):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator += ((${data_in_type.referencedType.typeName} (*)${shapeStr})${data_in})${accessStr};

% for i in range(len(axes)):
}
% endfor
% if keepdims:
*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) ((${data_out}_accumulator + ${data_out}_sgn*(${reduceLength}>>1)) / ${reduceLength} + ${output_offset});
% else:
<%

import numpy as np

if (np.log2(reduceLength) - int(np.log2(reduceLength))) == 0:
    shift = int(np.log2(reduceLength))
%>
% if shift is not None:
*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) (((${data_out}_accumulator + (1<<(${shift}-1))) >> ${shift}) + ${output_offset});
% else:
int8_t ${data_out}_sgn = 0;
${data_out}_sgn = -(${data_out}_accumulator<0) + (${data_out}_accumulator >= 0);
*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) ((${data_out}_accumulator + ${data_out}_sgn*(${reduceLength}>>1)) / ${reduceLength} + ${output_offset});
% endif
% endif
% for i in range(len(restDims)):
}
% endfor
END_SINGLE_CORE
""")
