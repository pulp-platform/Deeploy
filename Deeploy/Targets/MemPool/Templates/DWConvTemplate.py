# ----------------------------------------------------------------------
#
# File: DWConvTemplate.py
#
# Last edited: 09.01.2023
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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _DWConv2D_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels // 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        return ctxt, operatorRepresentation, []


MemPoolParallel1DTemplate = _DWConv2D_Template("""
<%
batchOffsetIn = ch_im_in  * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_y
%>

// 1D Depth-Wise Conv Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    DWConv2d_parallel_s${data_in_type.referencedType.typeWidth}_NCHW(
        ref_${data_out}_${data_in}, ${ch_im_in}, 1, ${dim_im_in_y},
        ${weight}, 1, ${dim_kernel_y},
        1, ${stride_y},
        ref_${data_out}_${data_out}, ${input_offset}, ${output_offset},
        core_id,
        numThreads
    );
    ref_${data_out}_${data_in} += ${batchOffsetIn};
    ref_${data_out}_${data_out} += ${batchOffsetOut};
}
mempool_barrier(numThreads);
""")

MemPoolParallel2DTemplate = _DWConv2D_Template("""
<%
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D Depth-Wise Conv Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    DWConv2d_parallel_s${data_in_type.referencedType.typeWidth}_NCHW(
        ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},
        ${weight}, ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${data_out}_${data_out}, ${input_offset}, ${output_offset},
        core_id,
        numThreads
    );
    ref_${data_out}_${data_in} += ${batchOffsetIn};
    ref_${data_out}_${data_out} += ${batchOffsetOut};
}
mempool_barrier(numThreads);
""")
