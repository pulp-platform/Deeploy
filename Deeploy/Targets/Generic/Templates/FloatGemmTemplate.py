# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// GEMM (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${A_type.typeName} ref_${nodeName}_${A} = ${A};
    ${B_type.typeName} ref_${nodeName}_${B} = ${B};
    ${C_type.typeName} ref_${nodeName}_${C} = ${C};
    ${data_out_type.typeName} ref_${nodeName}_${data_out} = ${data_out};

    for(uint32_t i=0; i<${batch}; i++){
        Gemm_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${C_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
            ref_${nodeName}_${A},
            ref_${nodeName}_${B},
            ref_${nodeName}_${C},
            ref_${nodeName}_${data_out},
            ${M},
            ${N},
            ${O},
            ${int(transA)},
            ${int(transB)}
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