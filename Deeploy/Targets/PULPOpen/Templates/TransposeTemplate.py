# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, _Template

_tileHeader = NodeTemplate("""
const uint32_t coreId = pi_core_id();

% for i in range(numDims):
uint16_t dimLen_${i} = <%text>${</%text>${dimLenPtr[i]}<%text>}</%text>;\n
% endfor
""")

_tileForLoop = NodeTemplate("""
const uint32_t baseChunk = dimLen_${i} / NUM_CORES;
const uint32_t leftover = dimLen_${i} - baseChunk * NUM_CORES;
const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
for(uint32_t i_${i} = offset; i_${i} < offset + chunk; i_${i}++ ) {
""")

_forLoop = NodeTemplate("""
for(uint32_t i_${i} = 0; i_${i} < dimLen_${i} ; i_${i}++){
""")


class PULPTransposeTemplate(NodeTemplate):

    def __init__(self, templateStr: str):
        self._indirectTemplate = _Template(templateStr)
        self.subTemplates = {}
        self.subTemplateGenerators = {}

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Layout (index strings + parallelDim) is computed in TransposeParser.
        # Here we only emit the per-dim loops and the tiling header.
        perm = operatorRepresentation['perm']
        parallelDim = operatorRepresentation['parallelDim']

        fRep = operatorRepresentation.copy()

        forLoops = []
        dimLenPtrs = []
        for idx, i in enumerate(perm):
            dimLenPtrs.append(f"dimLen_{idx}")
            if idx != parallelDim:
                forLoops.append(_forLoop.generate({"i": i, "dimLenPtr": f"dimLen_{i}"}))
            else:
                forLoops.append(_tileForLoop.generate({"i": i, "dimLenPtr": f"dimLen_{i}"}))

        fRep['forLoops'] = forLoops
        fRep['tileHeader'] = _tileHeader.generate({"numDims": len(perm), "dimLenPtr": dimLenPtrs})

        self.template = _Template(self._indirectTemplate.render(**fRep))

        return ctxt, operatorRepresentation, []


referenceTemplate = PULPTransposeTemplate("""
// Transpose ${data_in_shape} -> ${data_out_shape} (Name: ${nodeName}, Op: ${nodeOp})
${tileHeader}
// RW: GCC Segmentation fault
${data_in_type.referencedType.typeName} (*src)${shapeStr} = (${data_in_type.referencedType.typeName} (*)${shapeStr})<%text>${data_in}</%text>;
${data_in_type.referencedType.typeName} (*dst)${outShapeStr} = (${data_in_type.referencedType.typeName} (*)${outShapeStr})<%text>${data_out}</%text>;
% for idx, i in enumerate(perm):
${forLoops[idx]}
% endfor
dst${outAccessStr} = src${accessStr};
% for idx, i in enumerate(perm):
}
% endfor
""")
