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


class _GemmTemplate(NodeTemplate):

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

        if hasattr(A, "_signed") and hasattr(A, "nLevels"):
            operatorRepresentation['A_offset'] = (A._signed == 0) * int(A.nLevels / 2)
        if hasattr(B, "_signed") and hasattr(B, "nLevels"):
            operatorRepresentation['B_offset'] = (B._signed == 0) * int(B.nLevels / 2)
        if hasattr(C, "_signed") and hasattr(C, "nLevels"):
            operatorRepresentation['C_offset'] = -(C._signed == 0) * int(C.nLevels / 2)
        if hasattr(Y, "_signed") and hasattr(Y, "nLevels"):
            operatorRepresentation['Y_offset'] = -(Y._signed == 0) * int(Y.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _GemmTemplate("""
// GEMM (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${A_type.typeName} ref_${data_out}_${A} = ${A};
    ${B_type.typeName} ref_${data_out}_${B} = ${B};
    ${C_type.typeName} ref_${data_out}_${C} = ${C};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for(uint32_t i=0;i<${batch};i++){
        Gemm_s${A_type.referencedType.typeWidth}_s${B_type.referencedType.typeWidth}_s${C_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(
            ref_${data_out}_${A},
            ref_${data_out}_${B},
            ref_${data_out}_${C},
            ref_${data_out}_${data_out},
            ${M},
            ${N},
            ${O},
            ${alpha},
            ${beta},
            ${transA},
            ${transB},
            ${A_offset},
            ${B_offset},
            ${C_offset},
            ${Y_offset}
        );

        ref_${data_out}_${A} += ${M} * ${N};
        ref_${data_out}_${B} += ${N} * ${O};
        ref_${data_out}_${C} += ${M} * ${O};
        ref_${data_out}_${data_out} += ${M} * ${O};
    }
END_SINGLE_CORE
""")
