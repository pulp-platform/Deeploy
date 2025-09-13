# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

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

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
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
