# ----------------------------------------------------------------------
#
# File: GemmTemplate.py
#
# Last edited: 16.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate, OperatorRepresentation


class _GemmTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['C'])
        Y = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['A_offset'] = (A._signed == 0) * int(A.nLevels / 2)
        operatorRepresentation['B_offset'] = (B._signed == 0) * int(B.nLevels / 2)
        operatorRepresentation['C_offset'] = -(C._signed == 0) * int(C.nLevels / 2)
        operatorRepresentation['Y_offset'] = -(Y._signed == 0) * int(Y.nLevels / 2)

        # import ipdb; ipdb.set_trace()
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
            names += [name]
            ctxt.hoistTransientBuffer(name, size)
            operatorRepresentation['ctxtBuffer_A'] = ctxt._mangle(name)
        else:
            operatorRepresentation['ctxtBuffer_A'] = operatorRepresentation['A']

        size = operatorRepresentation['N'] * operatorRepresentation['O'] * (B._type.referencedType.typeWidth // 8)
        name = operatorRepresentation['nodeName'] + f"_buffer_B"
        operatorRepresentation['ctxtBuffer_B_size'] = size
        if isinstance(B, ConstantBuffer):
            names += [name]
            ctxt.hoistTransientBuffer(name, size)
            operatorRepresentation['ctxtBuffer_B'] = ctxt._mangle(name)
        else:
            operatorRepresentation['ctxtBuffer_B'] = operatorRepresentation['B']

        size = operatorRepresentation['M'] * operatorRepresentation['O'] * (C._type.referencedType.typeWidth // 8)
        name = operatorRepresentation['nodeName'] + f"_buffer_C"
        operatorRepresentation['ctxtBuffer_C_size'] = size
        if isinstance(C, ConstantBuffer):
            names += [name]
            ctxt.hoistTransientBuffer(name, size)
            operatorRepresentation['ctxtBuffer_C'] = ctxt._mangle(name)
        else:
            operatorRepresentation['ctxtBuffer_C'] = operatorRepresentation['C']

        return ctxt, operatorRepresentation, names


MemPoolParallelTemplate = _GemmTemplate("""
// GEMM Parallel (Name: ${nodeName}, Op: ${nodeOp})
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
    Gemm_parallel_s${A_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${C},
        ref_${data_out}_${data_out},
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
        ${Y_offset},
        core_id,
        numThreads
    );

    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
}
mempool_barrier(numThreads);
""")
