# ----------------------------------------------------------------------
#
# File: RequantShiftTemplate.py
#
# Last edited: 24.04.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RequantShiftTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        # operatorRepresentation['input_offset'] = (data_in._signed == 0) * operatorRepresentation['n_levels']//2
        # operatorRepresentation['output_offset'] = -(data_out._signed == 0) * operatorRepresentation['n_levels']//2
        operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = -(data_out._signed == 0) * operatorRepresentation['n_levels'] // 2

        operatorRepresentation['output_min'] = -(operatorRepresentation['n_levels'] // 2)
        operatorRepresentation['output_max'] = (operatorRepresentation['n_levels'] // 2) - 1

        return ctxt, operatorRepresentation, []


MemPoolParallelTemplate = _RequantShiftTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D
%>

// RequantShift Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);
% if channels_first:
    %if output_min==-128 and output_max==127 and data_in_type.referencedType.typeWidth==32 and data_out_type.referencedType.typeWidth==8 and size%4==0:
        RequantShift_unrolled_1x4_parallel_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NCHW(
            ${data_in},
            ${size},
            ${mul},
            ${add},
            ${data_out},
            ${log2Dstring},
            ${channel_width},
            ${input_offset},
            ${output_offset},
            1,
            core_id,
            numThreads);
    %else:
        RequantShift_parallel_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NCHW(
            ${data_in},
            ${size},
            ${mul},
            ${add},
            ${data_out},
            ${log2Dstring},
            ${channel_width},
            ${input_offset},
            ${output_offset},
            ${output_min},
            ${output_max},
            1,
            core_id,
            numThreads);
    %endif
% else:
    RequantShift_parallel_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NHWC(
        ${data_in},
        ${size},
        ${mul}, ${add},
        ${data_out},
        ${log2Dstring},
        ${channels},
        ${input_offset},
        ${output_offset},
        ${output_min},
        ${output_max},
        1,
        core_id,
        numThreads
    );
%endif
mempool_barrier(numThreads);
""")
