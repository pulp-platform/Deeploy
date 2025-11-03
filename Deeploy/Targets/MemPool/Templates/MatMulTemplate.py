# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _MatMulTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['A_offset'] = (A._signed == 0) * int(A.nLevels / 2)
        operatorRepresentation['B_offset'] = (B._signed == 0) * int(B.nLevels / 2)
        operatorRepresentation['offset_output'] = -(data_out._signed == 0) * int(data_out.nLevels / 2)

        # import ipdb; ipdb.set_trace()
        return ctxt, operatorRepresentation, []


MemPoolParallelTemplate = _MatMulTemplate("""
// MatMul Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);
${A_type.typeName} ref_${nodeName}_${A} = ${A};
${B_type.typeName} ref_${nodeName}_${B} = ${B};
${data_out_type.typeName} ref_${nodeName}_${data_out} = ${data_out};

for(uint32_t i=0;i<${batch};i++){
    MatMul_parallel_s${A_type.referencedType.typeWidth}(
        ref_${nodeName}_${A},
        ref_${nodeName}_${B},
        ref_${nodeName}_${data_out},
        ${M},
        ${N},
        ${O},
        ${A_offset}, ${B_offset}, ${offset_output},
        core_id,
        numThreads
    );

    ref_${nodeName}_${A} += ${M} * ${N};
    ref_${nodeName}_${B} += ${N} * ${O};
    ref_${nodeName}_${data_out} += ${M} * ${O};
}
mempool_barrier(numThreads);
""")
