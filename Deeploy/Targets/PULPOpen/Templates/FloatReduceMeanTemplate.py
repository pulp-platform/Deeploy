# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatReduceMeanTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatReduceMeanTemplate("""
## =============== Compute required variables ===============
## Compute the total number of elements being reduced in one axis
<%
reduceLength = 1

for i, axis in enumerate(axes):
    if axis < 0:
        axes[i] += len(data_in_shape)
    reduceLength = reduceLength * data_in_shape[axis]
%>

## Compute the remaining dimensions after reduction
## Order them for more efficient parallelization
<%
restDims = set(list(range(len(data_in_shape)))).difference(set(axes))
restDims = sorted(list(restDims), key=lambda x: data_in_shape[x])

dataSize = data_in_shape[restDims[-1]]
%>

## =============== Prepare shape and access strings ===============
## shapeStr is going to have the [d1][d2]... format
## accessStr is going to have the [i_0][i_1]... format
<%
    shapeStr = ''
    accessStr = ''

    data_out_str = '0'
    data_out_str_prod = 1
%>

% for idx, i in enumerate(data_in_shape[1:]):
<%
    shapeStr += '[' + str(i) + ']'
%>
% endfor

% for j in range(len(data_in_shape)):
<%
    accessStr += '[i_' + str(j) + ']'
%>
% endfor
                                             
% for k in sorted(restDims, reverse=True):
<%
    data_out_str += ' + i_' + str(k) + '*' + str(data_out_str_prod)
    data_out_str_prod = data_out_str_prod * data_in_shape[k]
%>
% endfor

## =============== Start of the actual template ===============
// ReduceMean (Name: ${nodeName}, Op: ${nodeOp})
## Get core information
uint32_t core_id = pi_core_id();
uint32_t log2Core = (uint32_t) LOG2(NUM_CORES);

## Split into chunks for each core
uint32_t chunk = (${dataSize}U >> log2Core) + ((${dataSize}U & (NUM_CORES - 1)) != 0);
uint32_t chunk_start = MIN(chunk * core_id, ${dataSize}U);
uint32_t chunk_stop = MIN(chunk_start + chunk, ${dataSize}U);

## Iterate through non-reduced dimensions
## Keep the last dimension for parallelization
% for i in list(restDims[:-1]):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++) {
% endfor
for(uint32_t i_${restDims[-1]} = chunk_start; i_${restDims[-1]}<chunk_stop; i_${restDims[-1]}++) {
## Initialize accumulator
uint32_t out_idx = ${data_out_str};
${data_out}[out_idx] = ${input_offset}*${reduceLength};

## Iterate through reduced dimensions and accumulate
% for i in list(axes):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++) {
% endfor
${data_out}[out_idx] += ((${data_in_type.referencedType.typeName} (*)${shapeStr})${data_in})${accessStr};
% for i in range(len(axes)):
}
% endfor

## Write back the mean value
% if keepdims:
${data_out}[out_idx] = (${data_out_type.referencedType.typeName}) (${data_out}[out_idx] / ${reduceLength} + ${output_offset});
% else:
${data_out}[out_idx] = (${data_out_type.referencedType.typeName}) (${data_out}[out_idx] / ${reduceLength});
% endif
% for i in range(len(restDims)):
}
% endfor
""")
