# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class FloatSoftmaxTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation["data_in"])
        operatorRepresentation["seq_len"] = data_in.shape[2]
        operatorRepresentation["input_samples"] = data_in.shape[-1]

        operatorRepresentation["kernelName"] = "Softmax_fp32"

        return ctxt, operatorRepresentation, []


FloatSoftmaxTemplateStr = r"""
    int32_t batch_size = ${size} / ${lastDimLength};
    int32_t compute_num = 1; //snrt_cluster_compute_core_num();
    int32_t ldI = compute_num * ${input_samples};
    int32_t batch_offset = ${seq_len} * ${input_samples};

    // JUNGVI: This implementation is broken and has memory leak.
    if (snrt_hartid() == 0){
        ${kernelName}(${data_in}, ${data_out}, ldI, batch_offset, batch_size, ${seq_len}, ${input_samples});
    }
"""

FloatSoftmax_Template = FloatSoftmaxTemplate(FloatSoftmaxTemplateStr)
