# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.CommonExtensions.NodeTemplate import GemmTemplate

referenceTemplate = GemmTemplate("""
// GEMM (Name: ${nodeName}, Op: ${nodeOp})
${A_type.typeName} ref_${data_out}_${A} = ${A};
${B_type.typeName} ref_${data_out}_${B} = ${B};
${C_type.typeName} ref_${data_out}_${C} = ${C};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0; i<${batch}; i++){
    PULP_Gemm_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${C_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${C},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ${int(transA)},
        ${int(transB)}
    );

    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${C} += ${M} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
}
""")