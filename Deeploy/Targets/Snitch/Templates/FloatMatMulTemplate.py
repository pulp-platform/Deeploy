# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Use snrt_cluster_core_idx() == 0 instead of BEGIN_SINGLE_CORE macro to avoid core_id dependency
referenceTemplate = NodeTemplate("""
// Matmul (Name: ${nodeName}, Op: ${nodeOp})
if (snrt_cluster_core_idx() == 0) {
    ${A_type.typeName} ref_${data_out}_${A} = ${A};
    ${B_type.typeName} ref_${data_out}_${B} = ${B};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for(uint32_t i=0; i<${batch}; i++){
        MatMul_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
            ref_${data_out}_${A},
            ref_${data_out}_${B},
            ref_${data_out}_${data_out},
            ${M},
            ${N},
            ${O}
        );

        ref_${data_out}_${A} += ${M} * ${N};
        ref_${data_out}_${B} += ${N} * ${O};
        ref_${data_out}_${data_out} += ${M} * ${O};
    }
}
""")
