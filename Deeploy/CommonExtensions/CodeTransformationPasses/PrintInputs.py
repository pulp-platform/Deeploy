# ----------------------------------------------------------------------
#
# File: PrintInput.py
#
# Last edited: 13.11.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

import re
from typing import Optional, Tuple

from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ConstantBuffer, ExecutionBlock, \
    NetworkContext, NodeTemplate, StructBuffer, TransientBuffer, _NoVerbosity

_DebugPrintTemplate = NodeTemplate("""
<%
import numpy as np
accessStr = ""
dimStr = ""
for idx, dim in enumerate(bufferShape):
    accessStr += "[" + f"print_iter_{idx}" + "]"
    if idx > 0:
        dimStr += "[" + f"{dim}" + "]"
%>
printf("${nodeName} ${bufferName}: ${bufferType.referencedType.typeName}, ${bufferShape}, %p\\n", ${bufferName});
% for idx, dim in enumerate(bufferShape):
printf("[");
for (int print_iter_${idx}=0; print_iter_${idx} < ${dim}; print_iter_${idx}++){
% endfor
printf("%*i,", 4, ((${bufferType.referencedType.typeName} (*)${dimStr})${bufferName})${accessStr});
% for dim in bufferShape:
}
printf("], \\n");
%endfor
""")


class PrintInputGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    def _getRepDict(self, ctxt: NetworkContext, ref: str, name: str):
        _buf = ctxt.lookup(ref)
        refbuf = _buf

        while hasattr(_buf, "_referenceName"):
            _buf = ctxt.lookup(_buf._referenceName)

        if isinstance(_buf, (TransientBuffer, ConstantBuffer, StructBuffer)):
            return None

        if name not in _buf._users:
            return None

        return {"bufferName": refbuf.name, "bufferType": _buf._type, "bufferShape": _buf.shape, "nodeName": name}

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        references = self.extractDynamicReferences(ctxt, executionBlock, True)

        for ref in references:
            refDict = self._getRepDict(ctxt, ref, name)
            if refDict is not None:
                executionBlock.addLeft(_DebugPrintTemplate, refDict)

        return ctxt, executionBlock


class MemoryAwareGeneration():

    def __init__(self, memoryHierarchyRegex: Optional[str] = None):
        super().__init__()
        if memoryHierarchyRegex is not None:
            self.regex = re.compile(memoryHierarchyRegex)
        else:
            self.regex = None

    def _matchesRegex(self, ctxt: NetworkContext, key: str) -> bool:
        _buffer = ctxt.lookup(key)

        if self.regex is None:
            return not hasattr(_buffer, "_memoryLevel")

        if not hasattr(_buffer, "_memoryLevel"):
            return False

        ret = self.regex.findall(ctxt.lookup(key)._memoryLevel)
        return ret != []


class MemoryAwarePrintInputGeneration(MemoryAwareGeneration, PrintInputGeneration):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        references = self.extractDynamicReferences(ctxt, executionBlock, True)

        filteredReferences = [ref for ref in references if self._matchesRegex(ctxt, ref)]

        for ref in filteredReferences:
            refDict = self._getRepDict(ctxt, ref, name)
            if refDict is not None:
                executionBlock.addLeft(_DebugPrintTemplate, refDict)

        return ctxt, executionBlock


class PrintOutputGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    def _getRepDict(self, ctxt: NetworkContext, ref: str, name: str):
        _buf = ctxt.lookup(ref)
        refbuf = _buf

        while hasattr(_buf, "_referenceName"):
            _buf = ctxt.lookup(_buf._referenceName)

        if isinstance(_buf, (TransientBuffer, ConstantBuffer, StructBuffer)):
            return None

        if name in _buf._users:
            return None

        if _buf._users == [] and not ctxt.is_global(_buf.name):
            return None

        return {"bufferName": refbuf.name, "bufferType": _buf._type, "bufferShape": _buf.shape, "nodeName": name}

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        references = self.extractDynamicReferences(ctxt, executionBlock, True)

        for ref in references:
            rep = self._getRepDict(ctxt, ref, name)
            if rep is not None:
                executionBlock.addRight(_DebugPrintTemplate, rep)

        return ctxt, executionBlock


class MemoryAwarePrintOutputGeneration(MemoryAwareGeneration, PrintOutputGeneration):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        references = self.extractDynamicReferences(ctxt, executionBlock, True)

        filteredReferences = [ref for ref in references if self._matchesRegex(ctxt, ref)]

        for ref in filteredReferences:
            refDict = self._getRepDict(ctxt, ref, name)
            if refDict is not None:
                executionBlock.addRight(_DebugPrintTemplate, refDict)

        return ctxt, executionBlock


class PrintConstantGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    def _getRepDict(self, ctxt: NetworkContext, ref: str, name: str):
        _buf = ctxt.lookup(ref)
        refbuf = _buf

        while hasattr(_buf, "_referenceName"):
            _buf = ctxt.lookup(_buf._referenceName)

        if not isinstance(_buf, ConstantBuffer) or _buf._users == []:
            return None

        return {"bufferName": refbuf.name, "bufferType": _buf._type, "bufferShape": _buf.shape, "nodeName": name}

    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
              name: str) -> Tuple[NetworkContext, ExecutionBlock]:

        references = self.extractDynamicReferences(ctxt, executionBlock, True)

        for ref in references:
            rep = self._getRepDict(ctxt, ref, name)
            if rep is not None:
                executionBlock.addLeft(_DebugPrintTemplate, rep)

        return ctxt, executionBlock


class MemoryAwarePrintConstantGeneration(MemoryAwareGeneration, PrintConstantGeneration):

    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
              name: str) -> Tuple[NetworkContext, ExecutionBlock]:

        references = self.extractDynamicReferences(ctxt, executionBlock, True)

        filteredReferences = [ref for ref in references if self._matchesRegex(ctxt, ref)]

        for ref in filteredReferences:
            refDict = self._getRepDict(ctxt, ref, name)
            if refDict is not None:
                executionBlock.addLeft(_DebugPrintTemplate, refDict)

        return ctxt, executionBlock
