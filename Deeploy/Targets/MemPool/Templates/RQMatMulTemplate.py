# ----------------------------------------------------------------------
#
# File: RQMatMulTemplate.py
#
# Last edited: 02.05.2023
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


class _RQMatMulTemplate(NodeTemplate, OperatorRepresentation):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['A_offset'] = (A._signed == 0) * int(A.nLevels / 2)
        operatorRepresentation['B_offset'] = (B._signed == 0) * int(B.nLevels / 2)
        operatorRepresentation['offset_output'] = -(data_out._signed == 0) * int(data_out.nLevels / 2)

        operatorRepresentation['output_min'] = -(operatorRepresentation['n_levels'] // 2)
        operatorRepresentation['output_max'] = (operatorRepresentation['n_levels'] // 2) - 1

        MUL = ctxt.lookup(operatorRepresentation['mul'])
        # WIESEP: Per element quantization is not supported for RQMatMul
        if len(MUL.shape) == 4:
            operatorRepresentation['perChannelQuant'] = int(MUL.shape[1] != 1)
            operatorRepresentation['perRowQuant'] = int(MUL.shape[2] != 1)
        elif len(MUL.shape) == 3:
            operatorRepresentation['perChannelQuant'] = int(MUL.shape[0] != 1)
            operatorRepresentation['perRowQuant'] = int(MUL.shape[1] != 1)
        elif len(MUL.shape) == 2:
            operatorRepresentation['perChannelQuant'] = 0
            operatorRepresentation['perRowQuant'] = int(MUL.shape[0] != 1)
        elif len(MUL.shape) == 1:
            operatorRepresentation['perChannelQuant'] = 0
            operatorRepresentation['perRowQuant'] = 0

        # import ipdb; ipdb.set_trace()
        return ctxt, operatorRepresentation, []

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Allocate buffer in L1 if original data lives in L2 to speed up the calculation,
        # by first transferring it to L2 with the DMA.

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])

        names = []
        size = operatorRepresentation['M'] * operatorRepresentation['N']
        name = operatorRepresentation['nodeName'] + f"_buffer_A"
        operatorRepresentation['ctxtBuffer_A_size'] = size
        if isinstance(A, ConstantBuffer):
            names += [name]
            ctxt.hoistTransientBuffer(name, size)
            operatorRepresentation['ctxtBuffer_A'] = ctxt._mangle(name)
        else:
            operatorRepresentation['ctxtBuffer_A'] = operatorRepresentation['A']

        size = operatorRepresentation['N'] * operatorRepresentation['O']
        name = operatorRepresentation['nodeName'] + f"_buffer_B"
        operatorRepresentation['ctxtBuffer_B_size'] = size
        if isinstance(B, ConstantBuffer):
            names += [name]
            ctxt.hoistTransientBuffer(name, size)
            operatorRepresentation['ctxtBuffer_B'] = ctxt._mangle(name)
        else:
            operatorRepresentation['ctxtBuffer_B'] = operatorRepresentation['B']

        return ctxt, operatorRepresentation, names


MemPoolParallelTemplate = _RQMatMulTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D
%>

// RQMatMul Parallel (Name: ${nodeName}, Op: ${nodeOp})
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

%if ctxtBuffer_A != A or ctxtBuffer_B != B:
    mempool_barrier(numThreads);
%endif

${A_type.typeName} ref_${data_out}_${A} = ${ctxtBuffer_A};
${B_type.typeName} ref_${data_out}_${B} = ${ctxtBuffer_B};
${mul_type.typeName} ref_${mul} = ${mul};
${add_type.typeName} ref_${add} = ${add};
${data_out_type.typeName}  ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0;i<${batch};i++){
%if A_offset==0 and B_offset==0 and offset_output==0 and M%4==0 and N%4==0 and O%4==0:
    RQMatMul_unrolled_2x2_parallel_s${A_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ref_${mul},
        ref_${add},
        ${log2Dstring},
        1,
        ${perRowQuant},
        core_id,
        numThreads
    );
%elif M%4==0 and N%4==0 and O%4==0:
    RQMatMul_offset_unrolled_2x2_parallel_s${A_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ref_${mul},
        ref_${add},
        ${log2Dstring},
        1,
        ${perRowQuant},
        ${A_offset}, ${B_offset}, ${offset_output},
        core_id,
        numThreads
    );
%else:
    RQMatMul_parallel_s${A_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ref_${mul},
        ref_${add},
        ${log2Dstring},
        1,
        ${perRowQuant},
        ${A_offset}, ${B_offset}, ${offset_output},
        ${output_min},
        ${output_max},
        core_id,
        numThreads
    );
%endif

    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
%if perChannelQuant:
    ++ref_${mul};
    ++ref_${add};
%endif
}
mempool_barrier(numThreads);
""")
