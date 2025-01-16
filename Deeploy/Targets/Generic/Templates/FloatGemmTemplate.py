# ----------------------------------------------------------------------
#
# File: GemmTemplate.py.py
#
# Last edited: 05.01.2023
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


class _FloatGemmTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['C'])
        Y = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['A_offset'] = 0
        operatorRepresentation['B_offset'] = 0
        operatorRepresentation['C_offset'] = 0
        operatorRepresentation['Y_offset'] = 0

        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatGemmTemplate("""
// GEMM float (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${A_type.typeName} ref_${data_out}_${A} = ${A};
    ${B_type.typeName} ref_${data_out}_${B} = ${B};
    ${C_type.typeName} ref_${data_out}_${C} = ${C};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for(uint32_t i=0; i<${batch}; i++){
        for(uint32_t m=0; m<${M}; m++){
            for(uint32_t n=0; n<${O}; n++){
                ref_${data_out}_${data_out}[m* ${O} + n] = ref_${data_out}_${C}[m * ${O} + n];
                for(uint32_t k=0; k<${N}; k++){
                    ref_${data_out}_${data_out}[m* ${O} + n] += ref_${data_out}_${A}[m * ${N} + k] * ref_${data_out}_${B}[k * ${O} + n];
                }
            }
        }

        ref_${data_out}_${A} += ${M} * ${O};
        ref_${data_out}_${B} += ${O} * ${N};
        ref_${data_out}_${C} += ${M} * ${N};
        ref_${data_out}_${data_out} += ${M} * ${N};
    }
END_SINGLE_CORE
""")
