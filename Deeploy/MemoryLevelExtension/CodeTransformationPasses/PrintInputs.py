# ----------------------------------------------------------------------
#
# File: PrintInput.py
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

from Deeploy.CommonExtensions.CodeTransformationPasses.PrintInputs import PrintConstantGeneration, \
    PrintInputGeneration, PrintOutputGeneration, _DebugPrintTemplate
from Deeploy.DeeployTypes import CodeGenVerbosity, ExecutionBlock, NetworkContext, _NoVerbosity


class _MemoryAwareGeneration():
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


class MemoryAwarePrintInputGeneration(_MemoryAwareGeneration, PrintInputGeneration):
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


class MemoryAwarePrintOutputGeneration(_MemoryAwareGeneration, PrintOutputGeneration):
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


class MemoryAwarePrintConstantGeneration(_MemoryAwareGeneration, PrintConstantGeneration):
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
