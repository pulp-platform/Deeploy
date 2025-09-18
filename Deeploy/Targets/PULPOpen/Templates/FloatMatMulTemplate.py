# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Matmul with row parallelism (Name: ${nodeName}, Op: ${nodeOp})

for(uint32_t b=0; b<${batch}; b++) {
    ${A_type.typeName} batch_A = ${A} + b * ${M} * ${N};
    ${B_type.typeName} batch_B = ${B} + b * ${N} * ${O};
    ${data_out_type.typeName} batch_out = ${data_out} + b * ${M} * ${O};

    PULP_MatMul_fp32_fp32_fp32_unroll1x7(
        batch_A,
        batch_B,
        batch_out,
        ${M},
        ${N},
        ${O}
    );
}
""")