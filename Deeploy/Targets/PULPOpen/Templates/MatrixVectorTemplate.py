# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _PULPMatrixVectorTemplate(NodeTemplate):

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


referenceTemplate = _PULPMatrixVectorTemplate("""
// MatrixVec (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int16_t ${nodeName}_chunk = (${int(batch)} >> ${nodeName}_log2Core) + ((${int(batch)} & (NUM_CORES-1))!=0);
int16_t ${nodeName}_chunk_start = MIN(${nodeName}_chunk*${nodeName}_core_id, ${int(batch)});
int16_t ${nodeName}_chunk_stop = MIN(${nodeName}_chunk_start + ${nodeName}_chunk, ${int(batch)} + 1);

int8_t* ref_${nodeName}_${A} = ${A} + (${nodeName}_chunk_start * ${M}*${N});
% if W_batched:
int8_t* ref_${nodeName}_${B} = ${B} + (${nodeName}_chunk_start * ${O}*${N});
% else:
int8_t* ref_${nodeName}_${B} = ${B};
% endif

int8_t* ref_${nodeName}_${data_out} = ${data_out} + (${nodeName}_chunk_start * ${O}*${M});

for (uint32_t i=${nodeName}_chunk_start;i<${nodeName}_chunk_stop;i++){
     gemv_s8_s8_plp(ref_${nodeName}_${A}, NULL, ref_${nodeName}_${data_out}, ref_${nodeName}_${B}, ${mul}, ${C}, 1, ${log2D}, ${N}, ${O}, 1, 1);
     ref_${nodeName}_${A} += (${M}*${N});
     ref_${nodeName}_${data_out} += (${O}*${M});

% if W_batched:
     ref_${nodeName}_${B} += ${N} * ${O};
% endif

}
""")
