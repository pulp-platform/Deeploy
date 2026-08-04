# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Multi-core MatMul (scalar, no SSR): all compute cores enter, kernel handles work distribution internally.
# Framework adds snrt_is_compute_core() guard and barriers via SnitchCoreFilterPass/SnitchSynchCoresPass.
# Works regardless of where the operands live (L1/L2), so it is the default reference path.
referenceTemplate = NodeTemplate("""
// Matmul (Name: ${nodeName}, Op: ${nodeOp})
{
    ${A_type.typeName} ref_${data_out}_${A} = ${A};
    ${B_type.typeName} ref_${data_out}_${B} = ${B};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for(uint32_t i=0; i<${batch}; i++){
        matmul_fp32_opt(
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

# Multi-core MatMul with SSR + FREP acceleration.
# Requires operands to reside in TCDM/L1 (Snitch SSR can only stream from cluster
# memory), so this template is intended for the tiled flow where DMA stages tiles
# into L1 first. Each core gets M/compute_num rows; SSR DM0 streams A, DM1 streams
# B, and FREP repeats the 8-wide FMA block over the K (reduction) dimension.
ssrFrepTemplate = NodeTemplate("""
// Matmul SSR+FREP (Name: ${nodeName}, Op: ${nodeOp})
{
    ${A_type.typeName} ref_${data_out}_${A} = ${A};
    ${B_type.typeName} ref_${data_out}_${B} = ${B};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for(uint32_t i=0; i<${batch}; i++){
        matmul_fp32_ssr_frep_oparallel(
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
