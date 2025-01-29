# ----------------------------------------------------------------------
#
# File: iSoftmaxTemplate.py
#
# Last edited: 30.05.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

        # softmax_fp32_opt kernel isn't supported by the current compiler
        # operatorRepresentation["kernelName"] = "Softmax_fp32_opt"

        return ctxt, operatorRepresentation, []


FloatSoftmaxTemplateStr = r"""
    uint32_t batch_size = ${size} / ${lastDimLength};
    uint32_t compute_num = snrt_cluster_compute_core_num();
    int32_t ldI = compute_num * ${input_samples};
    int32_t batch_offset = ${seq_len} * ${input_samples};
                                       
    ${kernelName}(${data_in}, ${data_out}, ldI, batch_offset, batch_size, ${seq_len}, ${input_samples});
"""

FloatSoftmax_Template = FloatSoftmaxTemplate(FloatSoftmaxTemplateStr)
