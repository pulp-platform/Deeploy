# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
formatSpecifier = "%*i"
if "float" in bufferType.referencedType.typeName or "double" in bufferType.referencedType.typeName:
    formatSpecifier = "%*.6f"
%>
printf("${nodeName} ${bufferName}: ${bufferType.referencedType.typeName}, ${bufferShape}, %p\\n", ${bufferName});
% for idx, dim in enumerate(bufferShape):
printf("[");
for (int print_iter_${idx}=0; print_iter_${idx} < ${dim}; print_iter_${idx}++){
% endfor
printf("${formatSpecifier},", 4, ((${bufferType.referencedType.typeName} (*)${dimStr})${bufferName})${accessStr});
% for dim in bufferShape:
}
printf("], \\n");
%endfor
""")


class PrintInputGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):
    """
    Code transformation pass for generating debug print statements for input tensors.

    This class extends CodeTransformationPass to automatically insert debug
    printing code that displays the contents of input tensors before operation
    execution. It's useful for debugging, verification, and analysis of neural
    network operations by showing the actual data values being processed.

    The generated print statements include tensor metadata (name, type, shape,
    memory address) and formatted tensor contents with proper indexing for
    multi-dimensional arrays.

    Notes
    -----
    This transformation only processes tensors that are actual inputs to the
    operation (not transient, constant, or struct buffers) and that have the
    operation in their user list. The printing is added before the operation
    execution.

    The generated C code uses nested loops to iterate through all tensor
    dimensions and prints values with appropriate formatting based on the
    data type (integer vs floating-point).
    """

    def _getRepDict(self, ctxt: NetworkContext, ref: str, name: str):
        """
        Create representation dictionary for a tensor reference.

        Analyzes a tensor reference to determine if it should be printed and
        creates the necessary template variables for code generation.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer information.
        ref : str
            The reference name of the tensor to analyze.
        name : str
            The name of the operation for filtering tensors.

        Returns
        -------
        dict or None
            A dictionary containing template variables if the tensor should
            be printed, None otherwise. The dictionary includes:
            - bufferName: The name of the buffer reference
            - bufferType: The data type of the buffer
            - bufferShape: The shape/dimensions of the buffer
            - nodeName: The operation name

        Notes
        -----
        Returns None for:
        - TransientBuffer, ConstantBuffer, or StructBuffer instances
        - Tensors that don't have the operation in their user list
        """
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
        """
        Apply input tensor printing transformation to an execution block.

        Analyzes all dynamic references in the execution block and adds debug
        print statements for input tensors before the operation execution.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer and type information.
        executionBlock : ExecutionBlock
            The execution block to instrument with input printing code.
        name : str
            The name of the operation being instrumented, used for filtering
            which tensors are considered inputs.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Default is _NoVerbosity.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The unchanged network context
            - The modified execution block with input print statements added

        Notes
        -----
        The transformation:
        1. Extracts all dynamic references from the execution block
        2. Filters for tensors that are inputs to this operation
        3. Adds debug print statements before the operation execution
        4. Generates formatted output showing tensor metadata and contents
        """

        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGlobalReferences = True)

        for ref in references:
            refDict = self._getRepDict(ctxt, ref, name)
            if refDict is not None:
                executionBlock.addLeft(_DebugPrintTemplate, refDict)

        return ctxt, executionBlock


class MemoryAwareGeneration():
    """
    Base class for memory-aware debug printing transformations.

    This class provides memory hierarchy filtering functionality for debug
    printing transformations. It allows selective printing of tensors based
    on their memory level assignments, enabling focused debugging of specific
    memory regions in multi-level memory architectures.

    Parameters
    ----------
    memoryHierarchyRegex : str, optional
        A regular expression pattern to match against buffer memory levels.
        If None, only buffers without memory level annotations are included.

    Attributes
    ----------
    regex : re.Pattern or None
        Compiled regular expression for memory level matching, or None if
        no filtering is applied.

    Notes
    -----
    This class is designed to be used as a mixin with specific printing
    transformation classes. It provides the `_matchesRegex` method for
    filtering buffers based on their memory level assignments.

    The regex-based filtering enables fine-grained control over which
    memory levels are included in debug output, which is crucial for
    debugging complex memory hierarchies in embedded neural network
    deployments.
    """

    def __init__(self, memoryHierarchyRegex: Optional[str] = None):
        """
        Initialize the MemoryAwareGeneration base class.

        Parameters
        ----------
        memoryHierarchyRegex : str, optional
            A regular expression pattern to match against buffer memory levels.
            If None, only buffers without memory level annotations are included.
        """
        super().__init__()
        if memoryHierarchyRegex is not None:
            self.regex = re.compile(memoryHierarchyRegex)
        else:
            self.regex = None

    def _matchesRegex(self, ctxt: NetworkContext, key: str) -> bool:
        """
        Check if a buffer matches the memory hierarchy regex pattern.

        Determines whether a buffer should be included in debug output based
        on its memory level assignment and the configured regex pattern.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer information.
        key : str
            The buffer reference key to check.

        Returns
        -------
        bool
            True if the buffer matches the criteria and should be included
            in debug output, False otherwise.

        Notes
        -----
        Matching logic:
        - If no regex is configured: matches buffers without memory level
        - If regex is configured: matches buffers whose memory level
          matches the regex pattern
        - Buffers without memory level annotations don't match when
          a regex is configured
        """
        _buffer = ctxt.lookup(key)

        if self.regex is None:
            return not hasattr(_buffer, "_memoryLevel")

        if not hasattr(_buffer, "_memoryLevel"):
            return False

        ret = self.regex.findall(ctxt.lookup(key)._memoryLevel)
        return ret != []


class MemoryAwarePrintInputGeneration(MemoryAwareGeneration, PrintInputGeneration):
    """
    Memory-aware input tensor debug printing transformation.

    This class combines MemoryAwareGeneration and PrintInputGeneration to
    provide selective debug printing of input tensors based on their memory
    level assignments. It's particularly useful for debugging multi-level
    memory architectures where you want to focus on specific memory regions.

    The class inherits filtering capabilities from MemoryAwareGeneration and
    input printing logic from PrintInputGeneration, applying memory-based
    filtering before generating debug print statements.

    Notes
    -----
    This transformation is especially valuable in embedded neural network
    deployments with complex memory hierarchies (e.g., L1/L2/L3 cache levels,
    scratchpad memories, external DRAM) where debugging specific memory
    regions is crucial for performance optimization and correctness verification.
    """

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """
        Apply memory-aware input tensor printing transformation.

        Filters input tensors by memory level before adding debug print
        statements, enabling focused debugging of specific memory regions.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer and type information.
        executionBlock : ExecutionBlock
            The execution block to instrument with filtered input printing code.
        name : str
            The name of the operation being instrumented.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Default is _NoVerbosity.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The unchanged network context
            - The modified execution block with filtered input print statements

        Notes
        -----
        The transformation:
        1. Extracts all dynamic references from the execution block
        2. Filters references based on memory level regex matching
        3. Further filters for tensors that are inputs to this operation
        4. Adds debug print statements for qualifying tensors
        """

        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGlobalReferences = True)

        filteredReferences = [ref for ref in references if self._matchesRegex(ctxt, ref)]

        for ref in filteredReferences:
            refDict = self._getRepDict(ctxt, ref, name)
            if refDict is not None:
                executionBlock.addLeft(_DebugPrintTemplate, refDict)

        return ctxt, executionBlock


class PrintOutputGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):
    """
    Code transformation pass for generating debug print statements for output tensors.

    This class extends CodeTransformationPass to automatically insert debug
    printing code that displays the contents of output tensors after operation
    execution. It's useful for debugging, verification, and analysis of neural
    network operations by showing the actual data values produced.

    The class complements PrintInputGeneration by focusing on outputs rather
    than inputs, providing a complete view of data flow through operations.

    Notes
    -----
    This transformation only processes tensors that are actual outputs from
    the operation (not used by the current operation, but either used by
    other operations or global buffers). The printing is added after the
    operation execution.

    Output tensors are identified by checking that the operation is NOT in
    their user list, indicating the operation produces rather than consumes
    the tensor.
    """

    def _getRepDict(self, ctxt: NetworkContext, ref: str, name: str):
        """
        Create representation dictionary for an output tensor reference.

        Analyzes a tensor reference to determine if it's an output tensor
        that should be printed and creates the necessary template variables.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer information.
        ref : str
            The reference name of the tensor to analyze.
        name : str
            The name of the operation for filtering tensors.

        Returns
        -------
        dict or None
            A dictionary containing template variables if the tensor should
            be printed, None otherwise. The dictionary includes:
            - bufferName: The name of the buffer reference
            - bufferType: The data type of the buffer
            - bufferShape: The shape/dimensions of the buffer
            - nodeName: The operation name

        Notes
        -----
        Returns None for:
        - TransientBuffer, ConstantBuffer, or StructBuffer instances
        - Tensors that have the operation in their user list (inputs)
        - Unused local tensors (not global and no users)
        """
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
        """
        Apply output tensor printing transformation to an execution block.

        Analyzes all dynamic references in the execution block and adds debug
        print statements for output tensors after the operation execution.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer and type information.
        executionBlock : ExecutionBlock
            The execution block to instrument with output printing code.
        name : str
            The name of the operation being instrumented, used for filtering
            which tensors are considered outputs.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Default is _NoVerbosity.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The unchanged network context
            - The modified execution block with output print statements added

        Notes
        -----
        The transformation:
        1. Extracts all dynamic references from the execution block
        2. Filters for tensors that are outputs from this operation
        3. Adds debug print statements after the operation execution
        4. Generates formatted output showing tensor metadata and contents
        """

        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGlobalReferences = True)

        for ref in references:
            rep = self._getRepDict(ctxt, ref, name)
            if rep is not None:
                executionBlock.addRight(_DebugPrintTemplate, rep)

        return ctxt, executionBlock


class MemoryAwarePrintOutputGeneration(MemoryAwareGeneration, PrintOutputGeneration):
    """
    Memory-aware output tensor debug printing transformation.

    This class combines MemoryAwareGeneration and PrintOutputGeneration to
    provide selective debug printing of output tensors based on their memory
    level assignments. It enables focused debugging of output data in specific
    memory regions within multi-level memory architectures.

    The class inherits filtering capabilities from MemoryAwareGeneration and
    output printing logic from PrintOutputGeneration, applying memory-based
    filtering before generating debug print statements for output tensors.

    Notes
    -----
    This transformation is particularly valuable for verifying that output
    data is correctly written to the intended memory levels in complex
    memory hierarchies, and for debugging memory management issues in
    embedded neural network deployments.
    """

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """
        Apply memory-aware output tensor printing transformation.

        Filters output tensors by memory level before adding debug print
        statements, enabling focused debugging of specific memory regions.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer and type information.
        executionBlock : ExecutionBlock
            The execution block to instrument with filtered output printing code.
        name : str
            The name of the operation being instrumented.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Default is _NoVerbosity.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The unchanged network context
            - The modified execution block with filtered output print statements

        Notes
        -----
        The transformation:
        1. Extracts all dynamic references from the execution block
        2. Filters references based on memory level regex matching
        3. Further filters for tensors that are outputs from this operation
        4. Adds debug print statements for qualifying tensors after execution
        """

        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGlobalReferences = True)

        filteredReferences = [ref for ref in references if self._matchesRegex(ctxt, ref)]

        for ref in filteredReferences:
            refDict = self._getRepDict(ctxt, ref, name)
            if refDict is not None:
                executionBlock.addRight(_DebugPrintTemplate, refDict)

        return ctxt, executionBlock


class PrintConstantGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):
    """
    Code transformation pass for generating debug print statements for constant tensors.

    This class extends CodeTransformationPass to automatically insert debug
    printing code that displays the contents of constant tensors used by
    operations. It's useful for debugging, verification, and analysis of
    neural network weights, biases, and other constant parameters.

    Constant tensors represent model parameters and other immutable data
    that don't change during execution. Printing these values helps verify
    that the correct parameters are loaded and accessible during operation
    execution.

    Notes
    -----
    This transformation only processes ConstantBuffer instances that have
    users (are actually referenced by operations). The printing is added
    before the operation execution to show the constant values being used.

    This is particularly useful for debugging quantization issues, parameter
    loading problems, and weight/bias verification in neural networks.
    """

    def _getRepDict(self, ctxt: NetworkContext, ref: str, name: str):
        """
        Create representation dictionary for a constant tensor reference.

        Analyzes a tensor reference to determine if it's a constant tensor
        that should be printed and creates the necessary template variables.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer information.
        ref : str
            The reference name of the tensor to analyze.
        name : str
            The name of the operation (used for template generation).

        Returns
        -------
        dict or None
            A dictionary containing template variables if the tensor should
            be printed, None otherwise. The dictionary includes:
            - bufferName: The name of the buffer reference
            - bufferType: The data type of the buffer
            - bufferShape: The shape/dimensions of the buffer
            - nodeName: The operation name

        Notes
        -----
        Returns None for:
        - Non-ConstantBuffer instances
        - Constant buffers with no users (unused constants)
        """
        _buf = ctxt.lookup(ref)
        refbuf = _buf

        while hasattr(_buf, "_referenceName"):
            _buf = ctxt.lookup(_buf._referenceName)

        if not isinstance(_buf, ConstantBuffer) or _buf._users == []:
            return None

        return {"bufferName": refbuf.name, "bufferType": _buf._type, "bufferShape": _buf.shape, "nodeName": name}

    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
              name: str) -> Tuple[NetworkContext, ExecutionBlock]:
        """
        Apply constant tensor printing transformation to an execution block.

        Analyzes all dynamic references in the execution block and adds debug
        print statements for constant tensors before the operation execution.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer and type information.
        executionBlock : ExecutionBlock
            The execution block to instrument with constant printing code.
        name : str
            The name of the operation being instrumented.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Default is _NoVerbosity.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The unchanged network context
            - The modified execution block with constant print statements added

        Notes
        -----
        The transformation:
        1. Extracts all dynamic references from the execution block
        2. Filters for constant buffers that have users
        3. Adds debug print statements before the operation execution
        4. Generates formatted output showing constant tensor metadata and values
        """

        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGlobalReferences = True)

        for ref in references:
            rep = self._getRepDict(ctxt, ref, name)
            if rep is not None:
                executionBlock.addLeft(_DebugPrintTemplate, rep)

        return ctxt, executionBlock


class MemoryAwarePrintConstantGeneration(MemoryAwareGeneration, PrintConstantGeneration):
    """
    Memory-aware constant tensor debug printing transformation.

    This class combines MemoryAwareGeneration and PrintConstantGeneration to
    provide selective debug printing of constant tensors based on their memory
    level assignments. It enables focused debugging of constant data (weights,
    biases, parameters) in specific memory regions within multi-level memory
    architectures.

    The class inherits filtering capabilities from MemoryAwareGeneration and
    constant printing logic from PrintConstantGeneration, applying memory-based
    filtering before generating debug print statements for constant tensors.

    Notes
    -----
    This transformation is particularly valuable for:
    - Verifying parameter placement in specific memory levels
    - Debugging weight loading and quantization in embedded deployments
    - Analyzing memory usage patterns for constant data
    - Troubleshooting parameter access issues in complex memory hierarchies

    It's especially useful in scenarios where different constant tensors
    are placed in different memory levels for performance optimization.
    """

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """
        Apply memory-aware constant tensor printing transformation.

        Filters constant tensors by memory level before adding debug print
        statements, enabling focused debugging of parameters in specific
        memory regions.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer and type information.
        executionBlock : ExecutionBlock
            The execution block to instrument with filtered constant printing code.
        name : str
            The name of the operation being instrumented.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Default is _NoVerbosity.
            This parameter is currently unused by the implementation.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The unchanged network context
            - The modified execution block with filtered constant print statements

        Notes
        -----
        The transformation:
        1. Extracts all dynamic references from the execution block
        2. Filters references based on memory level regex matching
        3. Further filters for constant buffers that have users
        4. Adds debug print statements for qualifying constant tensors
        """

        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGlobalReferences = True)

        filteredReferences = [ref for ref in references if self._matchesRegex(ctxt, ref)]

        for ref in filteredReferences:
            refDict = self._getRepDict(ctxt, ref, name)
            if refDict is not None:
                executionBlock.addLeft(_DebugPrintTemplate, refDict)

        return ctxt, executionBlock
