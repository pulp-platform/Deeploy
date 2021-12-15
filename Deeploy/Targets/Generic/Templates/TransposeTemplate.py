# ----------------------------------------------------------------------
#
# File: TransposeTemplate.py
#
# Last edited: 28.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Transpose ${data_in_shape} -> ${data_out_shape} (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
${data_out_type.typeName} dummy_${data_out} = ${data_out};
<%
    dimStr = ''
    accessStr = ''
    shapeStr = ''
    for dim in data_in_shape:
        dimStr += '['+str(dim)+']'
%>
% for idx, i in enumerate(perm[:-1]):
<%
    shapeStr += '['+str(data_in_shape[idx+1])+']'
%>
% endfor
% for idx, i in enumerate(perm):
<%
    shape = data_out_shape[idx]
    accessStr += '[i_'+str(idx)+']'
%>
for(uint32_t i_${i} = 0; i_${i}<${shape}; i_${i}++){
% endfor
*dummy_${data_out}++ = ((${data_in_type.referencedType.typeName} (*)${shapeStr})${data_in})${accessStr};
% for idx, i in enumerate(perm):
}
% endfor
END_SINGLE_CORE
""")
