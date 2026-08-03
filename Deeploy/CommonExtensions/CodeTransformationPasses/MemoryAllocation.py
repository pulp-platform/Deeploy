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
    """Template for allocating an argument struct."""

    def __init__(self, templateStr: str, bufferName: str):
        """Initialize the argument struct allocation template.

        Parameters
        ----------
        templateStr : str
            The template string.
        bufferName : str
            The name of the buffer.
        """
        super().__init__(templateStr)
        self.bufferName = bufferName


_stackAllocateTemplate = partial(
    _ArgStructAllocateTemplate,
    templateStr = "${structDict.typeName} ${name} = (${structDict.typeName}) ${str(structDict)};")


class ArgumentStructGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):
    """A code transformation pass that generates a struct for function arguments."""

    def __init__(self):
        super().__init__()

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """Apply the argument struct generation transformation.

        This transformation generates a struct for the function arguments. It allocates
        memory for the struct and initializes its fields.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context.
        executionBlock : ExecutionBlock
            The execution block.
        name : str
            The name of the argument.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            The transformed network context and execution block.
        """

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
    """A code transformation pass that manages memory allocation and deallocation for buffers.

    This pass is responsible for ensuring that memory is allocated for buffers when they are needed and deallocated when
    they are no longer in use."""

    def __init__(self, memoryLevelRegex: Optional[str] = None):
        """Initialize the memory management generation pass.

        Parameters
        ----------
        memoryLevelRegex : str, optional
            A regular expression to match memory levels.
        """
        super().__init__()
        if memoryLevelRegex is not None:
            self.regex = re.compile(memoryLevelRegex)
        else:
            self.regex = None

    def is_memory_level(self, buffer: VariableBuffer) -> bool:
        """Check if the given buffer is a memory level buffer.

        Parameters
        ----------
        buffer : VariableBuffer
            The buffer to check.

        Returns
        -------
        bool
            True if the buffer is a memory level buffer, False otherwise.
        """
        if self.regex is None:
            return not hasattr(buffer, "_memoryLevel")
        else:
            return hasattr(buffer, "_memoryLevel") and self.regex.fullmatch(buffer._memoryLevel) is not None

    @staticmethod
    def is_final_input(buffer: VariableBuffer, nodeName: str) -> bool:
        """Check if the given buffer is a final input buffer.

        Parameters
        ----------
        buffer : VariableBuffer
            The buffer to check.
        nodeName : str
            The name of the node to check against.

        Returns
        -------
        bool
            True if the buffer is a final input buffer, False otherwise.
        """
        return not isinstance(buffer, (StructBuffer, TransientBuffer)) and \
            len(buffer._users) > 0 and nodeName == buffer._users[-1]

    @staticmethod
    def is_output(buffer: VariableBuffer, nodeName: str) -> bool:
        """Check if the given buffer is an output buffer.

        """
        return not isinstance(buffer, (StructBuffer, TransientBuffer)) and nodeName not in buffer._users

    @staticmethod
    def is_transient(buffer: VariableBuffer, nodeName: str) -> bool:
        """Check if the given buffer is a transient buffer.

        Parameters
        ----------
        buffer : VariableBuffer
            The buffer to check.
        nodeName : str
            The name of the node to check against.

        Returns
        -------
        bool
            True if the buffer is a transient buffer, False otherwise.
        """
        return isinstance(buffer, TransientBuffer) and nodeName in buffer._users

    @staticmethod
    def topologicallySortBuffers(buffers: List[VariableBuffer]) -> List[VariableBuffer]:
        """
        Topologically sorts a list of VariableBuffer objects based on their reference dependencies.

        This method iteratively identifies buffers that are not referenced by any other buffer in the list,
        adding them to the sorted result. Buffers that reference others (via _ReferenceBuffer and _referenceName)
        are deferred until their dependencies are resolved. The process continues until all buffers are sorted,
        or a circular reference is detected (which raises an assertion error).

        The first buffers in the sorted list are those that do not have any dependencies, while the last buffers
        are those that are only referenced by others.

        Raises
        ------
        AssertionError
            If a circular reference is detected among the buffers, preventing a valid topological sort.

        Parameters
        ----------
        buffers : List[VariableBuffer]
            The list of buffers to sort.

        Returns
        -------
        List[VariableBuffer]
            The topologically sorted list of buffers.
        """
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
        """Apply the memory management generation transformation.

        This function is responsible for analyzing the memory usage of the given execution block
        and generating the necessary memory allocation and deallocation commands. It also takes care
        of managing the lifetimes of the buffers involved and ensuring that they are properly released
        when no longer needed.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context to use.
        executionBlock : ExecutionBlock
            The execution block to analyze.
        name : str
            The name of the node to check against.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            The updated network context and execution block.
        """

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
        # Topological sorting is necessary to ensure that we allocate reference buffers before their dependents
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
    """A code transformation pass that implements a 'passthrough' memory management strategy.

    In the context of code generation and memory management, 'passthrough' means that this pass does not
    perform any actual allocation or deallocation of memory buffers. Instead, it simply marks buffers as
    live or dead based on their usage, without modifying the underlying memory state and eventually generating
    code that reflects these changes.
    """

    def __init__(self, memoryHierarchyRegex: Optional[str] = None):
        """Initialize the memory management passthrough pass.

        Args:
            memoryHierarchyRegex (Optional[str], optional): A regex pattern to match memory hierarchy.
            Defaults to None.
        """
        super().__init__(memoryHierarchyRegex)

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """Apply the memory management passthrough transformation.

        This function marks buffers as live or dead based on their usage, without performing any actual
        memory allocation or deallocation.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context.
        executionBlock : ExecutionBlock
            The execution block.
        name : str
            The name of the buffer.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Defaults to _NoVerbosity.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            The updated network context and execution block.
        """
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
