# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
from typing import List, Optional, Tuple

from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, StructBuffer, TransientBuffer, VariableBuffer, _NoVerbosity, _ReferenceBuffer


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

    def __init__(self, memoryLevelRegex: Optional[str] = None):
        super().__init__()
        if memoryLevelRegex is not None:
            self.regex = re.compile(memoryLevelRegex)
        else:
            self.regex = None

    def is_memory_level(self, buffer: VariableBuffer) -> bool:
        if self.regex is None:
            return not hasattr(buffer, "_memoryLevel")
        else:
            return hasattr(buffer, "_memoryLevel") and self.regex.fullmatch(buffer._memoryLevel) is not None

    @staticmethod
    def is_final_input(buffer: VariableBuffer, nodeName: str) -> bool:
        return not isinstance(buffer, (StructBuffer, TransientBuffer)) and \
            len(buffer._users) > 0 and nodeName == buffer._users[-1]

    @staticmethod
    def is_output(buffer: VariableBuffer, nodeName: str) -> bool:
        return not isinstance(buffer, (StructBuffer, TransientBuffer)) and nodeName not in buffer._users

    @staticmethod
    def is_transient(buffer: VariableBuffer, nodeName: str) -> bool:
        return isinstance(buffer, TransientBuffer) and nodeName in buffer._users

    @staticmethod
    def topologicallySortBuffers(buffers: List[VariableBuffer]) -> List[VariableBuffer]:
        sortedBuffers = []
        unsortedBufferNames = [buff.name for buff in buffers]
        lastLen = len(unsortedBufferNames)

        while len(unsortedBufferNames) > 0:
            for buffer in buffers:
                if isinstance(buffer, _ReferenceBuffer) and buffer._referenceName in unsortedBufferNames:
                    continue

                sortedBuffers.append(buffer)
                unsortedBufferNames.remove(buffer.name)

            assert len(
                unsortedBufferNames) != lastLen, f"Circular reference detected among buffers: {unsortedBufferNames}"
            lastLen = len(unsortedBufferNames)

        return sortedBuffers

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGlobalReferences = False)
        localBuffers = [ctxt.localObjects[ref] for ref in references]
        memoryLevelBuffers = [buff for buff in localBuffers if self.is_memory_level(buff)]

        transients = [buff for buff in memoryLevelBuffers if self.is_transient(buff, name)]
        outputs = [buff for buff in memoryLevelBuffers if self.is_output(buff, name)]
        inputs = [buff for buff in memoryLevelBuffers if self.is_final_input(buff, name)]

        # We have to allocate the output buffers, unless they are global
        for buffer in reversed(self.topologicallySortBuffers(outputs + transients)):
            assert buffer._live == False, f"Tried to allocate already live buffer {buffer.name}"
            buffer._live = True

            memoryLevel = "None" if not hasattr(buffer, "_memoryLevel") else buffer._memoryLevel
            if memoryLevel not in ctxt._dynamicSize:
                ctxt._dynamicSize[memoryLevel] = int(buffer.sizeInBytes)
            else:
                ctxt._dynamicSize[memoryLevel] += int(buffer.sizeInBytes)

            executionBlock.addLeft(buffer.allocTemplate, buffer._bufferRepresentation())

        for levels in ctxt._dynamicSize.keys():
            if levels not in ctxt._maxDynamicSize:
                ctxt._maxDynamicSize[levels] = max(0, ctxt._dynamicSize[levels])
            else:
                ctxt._maxDynamicSize[levels] = max(ctxt._maxDynamicSize.get(levels, 0), ctxt._dynamicSize[levels])

        for buffer in inputs + transients:
            assert buffer._live == True, f"Tried to deallocate already dead buffer {buffer.name}"
            buffer._live = False
            # Don't deallocate if it's an alias of a live buffer
            if not buffer.has_live_aliases(ctxt):
                memoryLevel = "None" if not hasattr(buffer, "_memoryLevel") else buffer._memoryLevel
                if memoryLevel not in ctxt._dynamicSize:
                    ctxt._dynamicSize[memoryLevel] = 0
                else:
                    ctxt._dynamicSize[memoryLevel] -= int(buffer.sizeInBytes)
                executionBlock.addRight(buffer.deallocTemplate, buffer._bufferRepresentation())

        return ctxt, executionBlock


class MemoryPassthroughGeneration(MemoryManagementGeneration):

    def __init__(self, memoryHierarchyRegex: Optional[str] = None):
        super().__init__(memoryHierarchyRegex)

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGlobalReferences = False)
        localBuffers = [ctxt.localObjects[ref] for ref in references]
        memoryLevelBuffers = [buff for buff in localBuffers if self.is_memory_level(buff)]

        transients = [buff for buff in memoryLevelBuffers if self.is_transient(buff, name)]
        outputs = [buff for buff in memoryLevelBuffers if self.is_output(buff, name)]
        inputs = [buff for buff in memoryLevelBuffers if self.is_final_input(buff, name)]

        for buffer in outputs + transients:
            assert buffer._live == False, f"Tried to allocate already live buffer {buffer.name}"

            memoryLevel = "None" if not hasattr(buffer, "_memoryLevel") else buffer._memoryLevel
            if memoryLevel not in ctxt._dynamicSize:
                ctxt._dynamicSize[memoryLevel] = int(buffer.sizeInBytes)
            else:
                ctxt._dynamicSize[memoryLevel] += int(buffer.sizeInBytes)

            buffer._live = True

        for levels in ctxt._dynamicSize.keys():
            if levels not in ctxt._maxDynamicSize:
                ctxt._maxDynamicSize[levels] = max(0, ctxt._dynamicSize[levels])
            else:
                ctxt._maxDynamicSize[levels] = max(ctxt._maxDynamicSize.get(levels, 0), ctxt._dynamicSize[levels])

        for buffer in inputs + transients:
            assert buffer._live == True, f"Tried to deallocate already dead buffer {buffer.name}"

            memoryLevel = "None" if not hasattr(buffer, "_memoryLevel") else buffer._memoryLevel
            if memoryLevel not in ctxt._dynamicSize:
                ctxt._dynamicSize[memoryLevel] = 0
            else:
                ctxt._dynamicSize[memoryLevel] -= int(buffer.sizeInBytes)

            buffer._live = False

        return ctxt, executionBlock
