# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _PULPTallGEMMTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        signedW = ctxt.lookup(operatorRepresentation['B'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['A'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []


referenceTemplate = _PULPTallGEMMTemplate("""
// TallGEMM (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int16_t ${nodeName}_chunk = (${int(M)} >> ${nodeName}_log2Core) + ((${int(M)} & (NUM_CORES-1))!=0);
int16_t ${nodeName}_chunk_start = MIN(${nodeName}_chunk*${nodeName}_core_id, ${int(M)});
int16_t ${nodeName}_chunk_stop = MIN(${nodeName}_chunk_start + ${nodeName}_chunk, ${int(M)} + 1);

int8_t* ref_${nodeName}_${A};
int8_t* ref_${nodeName}_${B};
int8_t* ref_${nodeName}_${data_out};

for(int b=0; b<${batch}; b++){

    for (uint32_t i=${nodeName}_chunk_start; i<${nodeName}_chunk_stop; i++){

        int8_t* ref_${nodeName}_${A} = ${A} + (b * ${M} * ${N}) + (i * ${N});
        % if W_batched:
        int8_t* ref_${nodeName}_${B} = ${B} + (b * ${N} * ${O});
        % else:
        int8_t* ref_${nodeName}_${B} = ${B};
        % endif
        int8_t* ref_${nodeName}_${data_out} = ${data_out} + (b * ${M} * ${O}) + (i * ${O});

        gemv_s8_s8_plp(ref_${nodeName}_${A}, NULL, ref_${nodeName}_${data_out}, ref_${nodeName}_${B}, ${mul}, ${C}, 1, ${log2D}, ${N}, ${O}, 1, 1);
    }
}
""")
