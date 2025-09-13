# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _MaxPool2DTemplate(NodeTemplate):

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


MemPoolParallelTemplate = _MaxPool2DTemplate("""
<%
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D MaxPool Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    MaxPool2d_parallel_s${data_in_type.referencedType.typeWidth}_NCHW(
        ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},
        ${dim_kernel_x}, ${dim_kernel_y}, ${stride_x}, ${stride_y},
        ref_${data_out}_${data_out}, ${input_offset}, ${output_offset},
        core_id,
        numThreads
    );
    ref_${data_out}_${data_in} += ${batchOffsetIn};
    ref_${data_out}_${data_out} += ${batchOffsetOut};
}
mempool_barrier(numThreads);
""")
