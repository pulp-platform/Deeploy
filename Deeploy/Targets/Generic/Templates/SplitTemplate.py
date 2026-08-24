# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _SplitTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # The chunk lengths are already encoded in the output shapes, so the
        # 'split' input is only needed at parse time and never deployed.
        # (inspired by ReshapeTemplate)
        if 'split' in operatorRepresentation.keys():
            splitBuffer = ctxt.lookup(operatorRepresentation['split'])
            splitBuffer._deploy = False
            splitBuffer._live = False

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        axis: int = operatorRepresentation['axis']
        input_shape: tuple[int, ...] = tuple(data_in.shape)
        type_size: int = data_in._type.referencedType.typeWidth // 8
        num_outputs: int = operatorRepresentation['num_outputs']

        # Along the split axis every chunk is contiguous, so one iteration per
        # element of the preceding dimensions is enough to cover the tensor.
        inner_size = int(np.prod(input_shape[axis + 1:])) * type_size
        operatorRepresentation['iterations'] = int(np.prod(input_shape[:axis]))
        operatorRepresentation['stride'] = int(np.prod(input_shape[axis:])) * type_size

        chunk_bytes = []
        chunk_offsets = []
        offset = 0
        for idx in range(num_outputs):
            data_out = ctxt.lookup(operatorRepresentation[f'data_out_{idx}'])
            num_bytes = data_out.shape[axis] * inner_size
            chunk_bytes.append(num_bytes)
            chunk_offsets.append(offset)
            offset += num_bytes

        assert offset == operatorRepresentation['stride'], \
            f"Split chunks cover {offset} bytes, but the split axis spans {operatorRepresentation['stride']} bytes!"

        operatorRepresentation['chunk_bytes'] = chunk_bytes
        operatorRepresentation['chunk_offsets'] = chunk_offsets

        return ctxt, operatorRepresentation, []


referenceTemplate = _SplitTemplate("""
// Split (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
for (uint32_t i = 0; i < ${iterations}; i++) {
% for idx in range(num_outputs):
memcpy((char*) ${pageargs['data_out_' + str(idx)]} + i * ${chunk_bytes[idx]},
       (char*) ${data_in} + i * ${stride} + ${chunk_offsets[idx]},
       ${chunk_bytes[idx]});
% endfor
}
END_SINGLE_CORE
""")
