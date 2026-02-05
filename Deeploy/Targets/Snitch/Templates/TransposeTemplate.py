# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, _Template

# Two-stage header: <%text>${</%text> escapes produce ${dimLen_N} template variables
# that survive the first render and get resolved during the second render
# (by operatorRepresentation in untiled mode, or TilingVariableReplacement in tiled mode)
_tileHeader = NodeTemplate("""
const uint32_t _core_idx = snrt_cluster_core_idx();
const uint32_t _core_num = snrt_cluster_compute_core_num();

% for i in range(numDims):
uint32_t dimLen_${i} = <%text>${</%text>${dimLenPtr[i]}<%text>}</%text>;
% endfor
""")

_tileForLoop = NodeTemplate("""
const uint32_t _baseChunk_${i} = dimLen_${i} / _core_num;
const uint32_t _leftover_${i} = dimLen_${i} - _baseChunk_${i} * _core_num;
const uint32_t _offset_${i} = _baseChunk_${i} * _core_idx + (_core_idx < _leftover_${i} ? _core_idx : _leftover_${i});
const uint32_t _chunk_${i} = _core_idx < _leftover_${i} ? _baseChunk_${i} + 1 : _baseChunk_${i};
for(uint32_t i_${i} = _offset_${i}; i_${i} < _offset_${i} + _chunk_${i}; i_${i}++) {
""")

_forLoop = NodeTemplate("""
for(uint32_t i_${i} = 0; i_${i} < dimLen_${i}; i_${i}++) {
""")


class SnitchTransposeTemplate(NodeTemplate):

    def __init__(self, templateStr: str):
        self._indirectTemplate = _Template(templateStr)
        self.subTemplates = {}
        self.subTemplateGenerators = {}

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        shapeStr = ""
        dimStr = ""
        accessStr = ""
        outAccessStr = ""
        outShapeStr = ""
        perm = operatorRepresentation['perm']
        data_in_shape = ctxt.lookup(operatorRepresentation['data_in']).shape
        data_out_shape = ctxt.lookup(operatorRepresentation['data_out']).shape

        for idx, i in enumerate(perm[:-1]):
            shapeStr += '[' + f"dimLen_{idx+1}" + ']'
            outShapeStr += '[' + f"dimLen_{perm[idx+1]}" + ']'

        for dim in data_in_shape:
            dimStr += '[' + str(dim) + ']'

        for idx, i in enumerate(perm):
            accessStr += '[i_' + str(idx) + ']'
            outAccessStr += '[i_' + str(i) + ']'

        fRep = operatorRepresentation.copy()

        fRep['shapeStr'] = shapeStr
        fRep['outShapeStr'] = outShapeStr
        fRep['outAccessStr'] = outAccessStr
        fRep['dimStr'] = dimStr
        fRep['accessStr'] = accessStr
        fRep['data_out_shape'] = data_out_shape

        # Select the best dimension to parallelize:
        # prefer dimensions >= 8 for good load balancing, otherwise pick the largest
        parallelDims = [idx for idx, dim in enumerate(data_out_shape) if dim >= 8]
        if len(parallelDims) > 0:
            parallelDim = parallelDims[0]
        else:
            parallelDim = data_out_shape.index(max(data_out_shape))

        forLoops = []
        dimLenPtrs = []
        for idx, i in enumerate(perm):
            operatorRepresentation[f"dimLen_{idx}"] = data_in_shape[idx]
            dimLenPtrs.append(f"dimLen_{idx}")
            if idx != parallelDim:
                forLoops.append(_forLoop.generate({"i": i}))
            else:
                forLoops.append(_tileForLoop.generate({"i": i}))

        fRep['forLoops'] = forLoops
        fRep['tileHeader'] = _tileHeader.generate({"numDims": len(perm), "dimLenPtr": dimLenPtrs})
        fRep['parallelDim'] = parallelDim

        self.template = _Template(self._indirectTemplate.render(**fRep))

        return ctxt, operatorRepresentation, []


referenceTemplate = SnitchTransposeTemplate("""
// Transpose ${data_in_shape} -> ${data_out_shape} (Name: ${nodeName}, Op: ${nodeOp})
if (snrt_is_compute_core()) {
${tileHeader}
% for idx, i in enumerate(perm):
${forLoops[idx]}
% endfor
((${data_in_type.referencedType.typeName} (*)${outShapeStr})<%text>${data_out}</%text>)${outAccessStr} = ((${data_in_type.referencedType.typeName} (*)${shapeStr})<%text>${data_in}</%text>)${accessStr};
% for idx, i in enumerate(perm):
}
% endfor
}
""")
