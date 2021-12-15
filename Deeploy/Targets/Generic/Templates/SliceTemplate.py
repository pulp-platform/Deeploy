# ----------------------------------------------------------------------
#
# File: SliceTemplate.py
#
# Last edited: 01.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _SliceTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Immediate-ify start
        startsBuffer = ctxt.lookup(operatorRepresentation['starts'])
        axesBuffer = ctxt.lookup(operatorRepresentation['axes'])
        endsBuffer = ctxt.lookup(operatorRepresentation['ends'])
        stepsBuffer = ctxt.lookup(operatorRepresentation['steps'])

        startsBuffer._deploy = False
        axesBuffer._deploy = False
        endsBuffer._deploy = False
        stepsBuffer._deploy = False

        operatorRepresentation['starts'] = startsBuffer.values
        operatorRepresentation['ends'] = endsBuffer.values
        operatorRepresentation['axes'] = axesBuffer.values
        operatorRepresentation['steps'] = stepsBuffer.values

        operatorRepresentation['data_in_size'] = np.prod(operatorRepresentation['data_in_shape'])

        return ctxt, operatorRepresentation, []


referenceTemplate = _SliceTemplate("""
// Slice (Name: ${nodeName}, Op: ${nodeOp})
<%
dimSteps = []
dimSteps.append(data_in_size//data_in_shape[0])
for dim in data_in_shape[1:]:
     dimSteps.append(dimSteps[-1]//dim)
%>
<%
transferSize = dimSteps[axes[-1]]
%>
<%
if axes[0] > 0:
    preAxes = list(range(axes[0]))
else:
    preAxes = []
%>

${data_out_type.referencedType.typeName}* ref_${data_out} = ${data_out};
% for axis in (list(preAxes) + list(axes)):
uint32_t ${data_out}_offset_${axis} = 0;
% endfor

% for axis, axisLen in zip(preAxes, list(data_in_shape)):
for(uint32_t i_${axis} = 0; i_${axis} < ${axisLen}; i_${axis}++){
% if axis == 0:
${data_out}_offset_0 =  ${dimSteps[axis]} * i_${axis};
% else:
${data_out}_offset_${axis} =  ${data_out}_offset_${axis-1} + ${dimSteps[axis]} * i_${axis};
% endif
% endfor
% for axis, start, end, step in zip(axes, starts, ends, steps):
for(uint32_t i_${axis} = ${start}; i_${axis} < ${end}; i_${axis} += ${step}){
% if axis == 0:
${data_out}_offset_0 =  ${dimSteps[axis]} * i_${axis};
% else:
${data_out}_offset_${axis} =  ${data_out}_offset_${axis-1} + ${dimSteps[axis]} * i_${axis};
% endif
% endfor
memcpy(ref_${data_out}, ${data_in} + ${data_out}_offset_${axis}, ${transferSize* data_out_type.referencedType.typeWidth//8});
ref_${data_out} += ${transferSize};
% for axis in range(axes[-1]+1):
}
% endfor
""")
