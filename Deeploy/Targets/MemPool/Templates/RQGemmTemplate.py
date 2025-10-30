# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Sequence, Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate, OperatorRepresentation


class _RQGemmTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['C'])
        Y = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['A_offset'] = (A._signed == 0) * int(A.nLevels / 2)
        operatorRepresentation['B_offset'] = (B._signed == 0) * int(B.nLevels / 2)
        operatorRepresentation['C_offset'] = -(C._signed == 0) * int(C.nLevels / 2)
        operatorRepresentation['Y_offset'] = -(Y._signed == 0) * int(Y.nLevels / 2)

        operatorRepresentation['output_min'] = -(operatorRepresentation['n_levels'] // 2)
        operatorRepresentation['output_max'] = (operatorRepresentation['n_levels'] // 2) - 1

        MUL = ctxt.lookup(operatorRepresentation['mul'])
        # WIESEP: Per element and per column quantization is not supported for RQGemm

        if len(MUL.shape) == 1:
            operatorRepresentation['perRowQuant'] = 0
        else:
            operatorRepresentation['perRowQuant'] = int(MUL.shape[-2] != 1)

        return ctxt, operatorRepresentation, []

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Allocate buffer in L1 if original data lives in L2 to speed up the calculation,
        # by first transferring it to L2 with the DMA.

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['C'])

        names = []
        size = operatorRepresentation['M'] * operatorRepresentation['N'] * (A._type.referencedType.typeWidth // 8)
        name = operatorRepresentation['nodeName'] + f"_buffer_A"
        operatorRepresentation['ctxtBuffer_A_size'] = size
        if isinstance(A, ConstantBuffer):
            bufferName = ctxt.hoistTransientBuffer(name, size)
            names += [bufferName]
            operatorRepresentation['ctxtBuffer_A'] = bufferName
        else:
            operatorRepresentation['ctxtBuffer_A'] = operatorRepresentation['A']

        size = operatorRepresentation['N'] * operatorRepresentation['O'] * (B._type.referencedType.typeWidth // 8)
        name = operatorRepresentation['nodeName'] + f"_buffer_B"
        operatorRepresentation['ctxtBuffer_B_size'] = size
        if isinstance(B, ConstantBuffer):
            bufferName = ctxt.hoistTransientBuffer(name, size)
            names += [bufferName]
            operatorRepresentation['ctxtBuffer_B'] = bufferName
        else:
            operatorRepresentation['ctxtBuffer_B'] = operatorRepresentation['B']

        size = operatorRepresentation['M'] * operatorRepresentation['O'] * (C._type.referencedType.typeWidth // 8)
        name = operatorRepresentation['nodeName'] + f"_buffer_C"
        operatorRepresentation['ctxtBuffer_C_size'] = size
        if isinstance(C, ConstantBuffer):
            bufferName = ctxt.hoistTransientBuffer(name, size)
            names += [bufferName]
            operatorRepresentation['ctxtBuffer_C'] = bufferName
        else:
            operatorRepresentation['ctxtBuffer_C'] = operatorRepresentation['C']

        return ctxt, operatorRepresentation, names

    def alignShapes(self, node: gs.Node) -> Tuple[List[Sequence[int]], List[Sequence[int]]]:
        inShapes, outShapes = [t.shape for t in node.inputs], [t.shape for t in node.outputs]
        # rqs bias
        inShapes[2] = outShapes[0][-2:]
        # rqs add
        inShapes[3] = (1,)
        # rqs mul
        inShapes[4] = (1,)
        return inShapes, outShapes


MemPoolParallelTemplate = _RQGemmTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D
%>

// RQGEMM Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);

%if ctxtBuffer_A != A:
// Fast copy data from L2 to L1
BEGIN_SINGLE_CORE
    #if USE_DMA
        dma_memcpy_blocking(${ctxtBuffer_A}, ${A}, ${ctxtBuffer_A_size});
    #else
        memcpy(${ctxtBuffer_A}, ${A}, ${ctxtBuffer_A_size});
    #endif
END_SINGLE_CORE
%endif

%if ctxtBuffer_B != B:
// Fast copy data from L2 to L1
BEGIN_SINGLE_CORE
    #if USE_DMA
        dma_memcpy_blocking(${ctxtBuffer_B}, ${B}, ${ctxtBuffer_B_size});
    #else
        memcpy(${ctxtBuffer_B}, ${B}, ${ctxtBuffer_B_size});
    #endif
END_SINGLE_CORE
%endif

%if ctxtBuffer_C != C:
// Fast copy data from L2 to L1
BEGIN_SINGLE_CORE
    #if USE_DMA
        dma_memcpy_blocking(${ctxtBuffer_C}, ${C}, ${ctxtBuffer_C_size});
    #else
        memcpy(${ctxtBuffer_C}, ${C}, ${ctxtBuffer_C_size});
    #endif
END_SINGLE_CORE
%endif

%if ctxtBuffer_A != A or ctxtBuffer_B != B or ctxtBuffer_C != C:
    mempool_barrier(numThreads);
%endif

${A_type.typeName} ref_${data_out}_${A} = ${ctxtBuffer_A};
${B_type.typeName} ref_${data_out}_${B} = ${ctxtBuffer_B};
${C_type.typeName} ref_${data_out}_${C} = ${ctxtBuffer_C};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0;i<${batch};i++){
%if M%4==0 and N%4==0 and O%4==0:
    RQGemm_offset_unrolled_2x2_parallel_s${A_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${C},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ${alpha},
        ${beta},
        ${int(transA)},
        ${int(transB)},
        ${mul},
        ${add},
        ${log2Dstring},
        1,
        ${perRowQuant},
        ${A_offset},
        ${B_offset},
        ${C_offset},
        ${Y_offset},
        core_id,
        numThreads
    );
%else:
    RQGemm_parallel_s${A_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${C},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ${alpha},
        ${beta},
        ${int(transA)},
        ${int(transB)},
        ${mul},
        ${add},
        ${log2Dstring},
        1,
        ${perRowQuant},
        ${A_offset},
        ${B_offset},
        ${C_offset},
        ${Y_offset},
        ${output_min},
        ${output_max},
        core_id,
        numThreads
    );
%endif

    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
}
mempool_barrier(numThreads);
""")
