# ----------------------------------------------------------------------
#
# File: MemoryAllocation.py
#
# Last edited: 12.06.2023
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
from functools import partial
from typing import List, Optional, Tuple

from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, StructBuffer, TransientBuffer, _NoVerbosity


class _ArgStructAllocateTemplate(NodeTemplate):

    def __init__(self, templateStr: str, bufferName: str):
        super().__init__(templateStr)
        self.bufferName = bufferName


_stackAllocateTemplate = partial(
    _ArgStructAllocateTemplate,
    templateStr = "${structDict.typeName} ${name} = (${structDict.typeName}) ${str(structDict)};")


class ArgumentStructGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    def __init__(self):
        super().__init__()

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        references = self.extractDynamicReferences(ctxt, executionBlock, True)
        buffers = [ctxt.lookup(key) for key in references]

        closureStructBufferNames = [
            codeSnippet.template.bufferName
            for codeSnippet in executionBlock.codeSnippets
            if isinstance(codeSnippet.template, _ArgStructAllocateTemplate)
        ]

        buffers = [buf for buf in buffers if buf.name not in closureStructBufferNames]

        for _buffer in buffers:
            if isinstance(_buffer, StructBuffer) and name in _buffer._users:
                executionBlock.addLeft(_stackAllocateTemplate(bufferName = _buffer.name),
                                       _buffer._bufferRepresentation())

        return ctxt, executionBlock


class MemoryManagementGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

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

    def _extractTransientBuffers(self, ctxt: NetworkContext, name: str) -> List[str]:
        names = []

        for key, _buffer in ctxt.localObjects.items():
            if isinstance(_buffer, TransientBuffer) and name in _buffer._users:
                names.append(key)

        filteredNames = [key for key in names if self._matchesRegex(ctxt, key)]

        return filteredNames

    def _getOutputNames(self, ctxt: NetworkContext, executionBlock: ExecutionBlock, name: str) -> List[str]:
        outputs = []
        references = self.extractDynamicReferences(ctxt, executionBlock, True)
        localKeys = [key for key in references if ctxt.is_local(key)]

        filteredKeys = [key for key in localKeys if self._matchesRegex(ctxt, key)]

        for key in filteredKeys:
            _buffer = ctxt.lookup(key)
            if isinstance(_buffer, (StructBuffer, TransientBuffer)):
                continue
            if name not in _buffer._users:
                outputs.append(_buffer.name)

        return list(dict.fromkeys(outputs))

    def _getFinalInputNames(self, ctxt: NetworkContext, executionBlock: ExecutionBlock, name: str) -> List[str]:
        inputs = []
        references = self.extractDynamicReferences(ctxt, executionBlock, True)
        localKeys = [key for key in references if ctxt.is_local(key)]

        filteredKeys = [key for key in localKeys if self._matchesRegex(ctxt, key)]

        for key in filteredKeys:
            _buffer = ctxt.lookup(key)
            if isinstance(_buffer, (StructBuffer, TransientBuffer)) or _buffer._users == []:
                continue
            if name == _buffer._users[-1]:
                inputs.append(_buffer.name)

        return list(dict.fromkeys(inputs))

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        outputNames = self._getOutputNames(ctxt, executionBlock, name)
        inputNames = self._getFinalInputNames(ctxt, executionBlock, name)
        transientBuffers = self._extractTransientBuffers(ctxt, name)

        # We have to allocate the output buffers, unless they are global

        for buffer in list(reversed(outputNames)) + transientBuffers:
            nb = ctxt.lookup(buffer)
            assert ctxt.localObjects[nb.name]._live == False, f"Tried to allocate already live buffer {nb.name}"
            ctxt.localObjects[nb.name]._live = True
            executionBlock.addLeft(nb.allocTemplate, nb._bufferRepresentation())

        for buffer in inputNames + transientBuffers:
            nb = ctxt.lookup(buffer)
            assert ctxt.localObjects[nb.name]._live == True, f"Tried to deallocate already dead buffer {nb.name}"
            ctxt.localObjects[nb.name]._live = False
            executionBlock.addRight(nb.deallocTemplate, nb._bufferRepresentation())

        return ctxt, executionBlock


class MemoryPassthroughGeneration(MemoryManagementGeneration, IntrospectiveCodeTransformationMixIn):

    def __init__(self, memoryHierarchyRegex: Optional[str] = None):
        super().__init__(memoryHierarchyRegex)

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        outputNames = self._getOutputNames(ctxt, executionBlock, name)
        inputNames = self._getFinalInputNames(ctxt, executionBlock, name)
        transientBuffers = self._extractTransientBuffers(ctxt, name)

        # We have to allocate the output buffers, unless they are global
        for buffer in outputNames + transientBuffers:
            nb = ctxt.lookup(buffer)

            assert ctxt.localObjects[nb.name]._live == False, f"Tried to allocate already live buffer {nb.name}"
            ctxt.localObjects[nb.name]._live = True

        for buffer in inputNames + transientBuffers:
            nb = ctxt.lookup(buffer)

            assert ctxt.localObjects[nb.name]._live == True, f"Tried to deallocate already dead buffer {nb.name}"
            ctxt.localObjects[nb.name]._live = False

        return ctxt, executionBlock
