# ----------------------------------------------------------------------
#
# File: TransposeTemplate.py
#
# Last edited: 28.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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
                forLoops.append(_forLoop.generate({"i": i, "dimLenPtr": f"dimLen_{i}"}))
            else:
                forLoops.append(_tileForLoop.generate({"i": i, "dimLenPtr": f"dimLen_{i}"}))

        fRep['forLoops'] = forLoops
        fRep['tileHeader'] = _tileHeader.generate({"numDims": len(perm), "dimLenPtr": dimLenPtrs})
        fRep['parallelDim'] = parallelDim

        self.template = _Template(self._indirectTemplate.render(**fRep))

        return ctxt, operatorRepresentation, []


referenceTemplate = PULPTransposeTemplate("""
// Transpose ${data_in_shape} -> ${data_out_shape} (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
${tileHeader}
% for idx, i in enumerate(perm):
${forLoops[idx]}
% endfor
((${data_in_type.referencedType.typeName} (*)${outShapeStr})<%text>${data_out}</%text>)${outAccessStr} = ((${data_in_type.referencedType.typeName} (*)${shapeStr})<%text>${data_in}</%text>)${accessStr};
% for idx, i in enumerate(perm):
}
% endfor
END_SINGLE_CORE
""")
