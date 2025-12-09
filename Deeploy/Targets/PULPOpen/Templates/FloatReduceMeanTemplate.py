# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatReduceMeanTemplate(NodeTemplate):
    '''
    WARNING: This version of parallelization is optimized for the TinyViT ReduceMean layers
    (49 elements in the reduced axis). Greater sizes of the reduced axis may benefit
    from different parallelization and tiling strategies.
    '''

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
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels / 2)

        for ax in range(len(operatorRepresentation['data_in_shape'])):
            if ax not in operatorRepresentation['axes']:
                _ = operatorRepresentation['dim_in_' + str(ax)]

        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatReduceMeanTemplate("""
## =============== Perform necessary precomputations ===============
<%
# Update input shape based on tiling
new_data_in_shape = data_in_shape.copy()

for i in range(len(new_data_in_shape)):
    if i not in axes:
        new_data_in_shape[i] = pageargs['dim_in_' + str(i)]

# Compute the total number of elements being reduced in one axis
reduceLength = 1
for i, axis in enumerate(axes):
    if axis < 0:
        axes[i] += len(data_in_shape)
    reduceLength = reduceLength * data_in_shape[axis]

# Compute the remaining dimensions after reduction
# Order them for more efficient parallelization
# (heuristically working on the largest non-tiled stride last,
# since it's impossible to get tiling information here)
restDims = list(set(list(range(len(data_in_shape)))).difference(set(axes)))
restDims = sorted(restDims, key=lambda x: data_in_shape[x])

dataSize = new_data_in_shape[restDims[-1]]

# =============== Prepare shape and access strings ===============
# shapeStr is going to have the [d1][d2]... format
# accessStr is going to have the [i_0][i_1]... format
shapeStr = ''
accessStr = ''

data_out_str = "0"
data_out_str_prod = "1"

for idx, i in enumerate(new_data_in_shape[1:]):
    if isinstance(i, str):
        shapeStr += '[*' + i + ']'
    else:
        shapeStr += '[' + str(i) + ']'

for j in range(len(data_in_shape)):
    accessStr += '[i_' + str(j) + ']'

for k in sorted(restDims, reverse=True):
    data_out_str += ' + i_' + str(k) + '*' + str(data_out_str_prod)
    if isinstance(new_data_in_shape[k], str):
        data_out_str_prod += "* *(" + new_data_in_shape[k] + ")"
    else:
        data_out_str_prod += "* " + str(new_data_in_shape[k])
%>

## =============== Start of the actual template ===============
// ReduceMean (Name: ${nodeName}, Op: ${nodeOp})
## Get core information
uint32_t core_id = pi_core_id();
uint32_t log2Core = (uint32_t) LOG2(NUM_CORES);

## Split into chunks for each core
% if isinstance(dataSize, str):
uint32_t chunk = (*(${dataSize}) >> log2Core) + ((*(${dataSize}) & (NUM_CORES - 1)) != 0);
uint32_t chunk_start = MIN(chunk * core_id, *(${dataSize}));
uint32_t chunk_stop = MIN(chunk_start + chunk, *(${dataSize}));
% else:
uint32_t chunk = (${dataSize}U >> log2Core) + ((${dataSize}U & (NUM_CORES - 1)) != 0);
uint32_t chunk_start = MIN(chunk * core_id, ${dataSize}U);
uint32_t chunk_stop = MIN(chunk_start + chunk, ${dataSize}U);
% endif

## Iterate through non-reduced dimensions
## Keep the last dimension for parallelization
% for i in list(restDims[:-1]):
% if isinstance(pageargs['dim_in_' + str(i)], str):
for(uint32_t i_${i} = 0; i_${i} < *${pageargs['dim_in_' + str(i)]}; i_${i}++) {
% else:
for(uint32_t i_${i} = 0; i_${i} < ${pageargs['dim_in_' + str(i)]}; i_${i}++) {
% endif
% endfor
for(uint32_t i_${restDims[-1]} = chunk_start; i_${restDims[-1]} < chunk_stop; i_${restDims[-1]}++) {
## Initialize accumulator
uint32_t out_idx = ${data_out_str};
${data_out}[out_idx] = ${input_offset}*${reduceLength};

## Iterate through reduced dimensions and accumulate
% for i in list(axes):
for(uint32_t i_${i} = 0; i_${i} < ${data_in_shape[i]}; i_${i}++) {
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
