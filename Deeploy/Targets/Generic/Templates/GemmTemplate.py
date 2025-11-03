# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
    ${A_type.typeName} ref_${nodeName}_${A} = ${A};
    ${B_type.typeName} ref_${nodeName}_${B} = ${B};
    ${C_type.typeName} ref_${nodeName}_${C} = ${C};
    ${data_out_type.typeName} ref_${nodeName}_${data_out} = ${data_out};

    for(uint32_t i=0;i<${batch};i++){
        Gemm_s${A_type.referencedType.typeWidth}_s${B_type.referencedType.typeWidth}_s${C_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(
            ref_${nodeName}_${A},
            ref_${nodeName}_${B},
            ref_${nodeName}_${C},
            ref_${nodeName}_${data_out},
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

        % if A_batched:
        ref_${nodeName}_${A} += ${M} * ${N};
        % endif

        % if B_batched:
        ref_${nodeName}_${B} += ${N} * ${O};
        % endif

        % if C_batched:
        ref_${nodeName}_${C} += ${M} * ${O};
        % endif

        ref_${nodeName}_${data_out} += ${M} * ${O};
    }
END_SINGLE_CORE
""")
