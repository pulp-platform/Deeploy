# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple, Type, Union

from Deeploy.AbstractDataTypes import Immediate, Pointer, PointerClass, Struct, StructClass, VoidType
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, StructBuffer, TransientBuffer, _NoVerbosity

# SCHEREMO: example template for a function closure call
_closureCallTemplate = NodeTemplate("""
// ${closureName} CLOSURE CALL
${closureName}(&${closureStructArgName});
""")

_closureTemplate = NodeTemplate("""
static void ${closureName}(void* ${closureName}_args){
// CLOSURE ARG CAST
% if len(closureStructArgs.value) > 0:
${closureStructArgs.typeName}* args = (${closureStructArgs.typeName}*) ${closureStructArgName};
% endif

% for argName, argType in closureStructArgs.value.items():
${argType.typeName} ${argName} = args->${argName};
% endfor

// CLOSURE FUNCTION CALL
${functionCall}

// CLOSURE ARG WRITEBACK
% if writeback:
% for argName, argType in closureStructArgs.value.items():
args->${argName} = ${argName};
% endfor
% endif
}
""")

_closureWriteBackTemplate = NodeTemplate("""
// CLOSURE ARG WRITEBACK
% for argName, argType in closureStructArgs.value.items():
${argName} = ${closureStructArgName}.${argName};
% endfor
""")

_closureStructDefTemplate = NodeTemplate("""
typedef struct ${closureStructArgs._typeDefRepr()} ${closureStructArgName}_t;
""")


class ClosureExecutionBlock(ExecutionBlock):
    """
    Execution block wrapper for closure-based code generation.

    This class extends ExecutionBlock to support closure-based code generation
    patterns, where functions are wrapped in closures with argument structures.
    It maintains a reference to the base execution block that contains the
    actual code to be wrapped.

    Notes
    -----
    This class is used in the closure generation process to maintain the
    relationship between the closure wrapper and the original execution block.
    """

    def __init__(self, nodeTemplate = None, closureBlock: Optional[ExecutionBlock] = None):
        """
        Initialize a ClosureExecutionBlock.

        Parameters
        ----------
        nodeTemplate : NodeTemplate, optional
            The node template for this execution block. Default is None.
        closureBlock : ExecutionBlock, optional
            The execution block to be wrapped in a closure. Default is None.
        """
        super().__init__(nodeTemplate)
        self.closureBlock = closureBlock

    @property
    def baseBlock(self):
        """
        Get the base execution block, unwrapping nested closures.

        Recursively unwraps ClosureExecutionBlock instances to find the
        underlying base execution block that contains the actual code.

        Returns
        -------
        ExecutionBlock
            The base execution block without closure wrappers.

        Notes
        -----
        This property handles nested closures by recursively calling
        baseBlock until a non-ClosureExecutionBlock is found.
        """
        if isinstance(self.closureBlock, ClosureExecutionBlock):
            return self.closureBlock.baseBlock
        return self.closureBlock


class ClosureGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):
    """
    Code transformation pass for generating function closures.

    This class transforms execution blocks into closure-based code patterns
    where functions are wrapped with argument structures. It generates the
    necessary struct definitions, closure functions, and call sites to
    enable closure-based execution patterns in generated code.


    Notes
    -----
    The closure generation process involves:
    1. Analyzing the execution block to identify dynamic references
    2. Creating a struct type to hold closure arguments
    3. Generating the closure function definition
    4. Replacing the original call with a closure call
    5. Optionally generating argument writeback code
    """

    closureStructArgType: Dict[str, Type[Union[Pointer, Immediate, Struct]]]
    closureStructArgs: Dict[str, Union[Pointer, Immediate, Struct]]

    def __init__(self,
                 closureCallTemplate: NodeTemplate = _closureCallTemplate,
                 closureSuffix = "_closure",
                 writeback: bool = True,
                 generateStruct: bool = True):
        """
        Initialize the ClosureGeneration transformation pass.

        Parameters
        ----------
        closureCallTemplate : NodeTemplate, optional
            Template for generating closure function calls. Default is the
            global _closureCallTemplate.
        closureSuffix : str, optional
            Suffix to append to closure function names. Default is "_closure".
        writeback : bool, optional
            Whether to generate writeback code for closure arguments.
            Default is True.
        generateStruct : bool, optional
            Whether to generate argument structure definitions. Default is True.
        """
        super().__init__()
        self.closureSuffix = closureSuffix
        self.closureTemplate = _closureTemplate
        self.closureCallTemplate = closureCallTemplate
        self.closureStructDefTemplate = _closureStructDefTemplate
        self.closureWriteBackTemplate = _closureWriteBackTemplate
        self.writeback = writeback
        self.generateStruct = generateStruct

    # Don't override this
    def _generateClosureStruct(self, ctxt: NetworkContext, executionBlock: ExecutionBlock):
        """
        Generate the closure argument structure.

        Analyzes the execution block to identify dynamic references and creates
        a struct type to hold all closure arguments. This struct will be used
        to pass arguments to the closure function.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer information.
        executionBlock : ExecutionBlock
            The execution block to analyze for dynamic references.

        Notes
        -----
        This method populates the following instance attributes:
        - closureStructArgType: The struct class type for closure arguments
        - closureStructArgs: The struct instance with argument mappings

        The method handles different buffer types:
        - TransientBuffer: Mapped to void pointers
        - StructBuffer: Excluded from closure arguments
        - Other buffers: Use their native types
        """

        # Add closure struct info to operatorRepresentation
        closureStructArgsType: Dict[str, Type[Union[Pointer, Immediate, Struct]]] = {}
        closureStruct: Dict[str, Union[Pointer, Immediate, Struct]] = {}
        makoDynamicReferences = self.extractDynamicReferences(ctxt, executionBlock, True)

        for arg in makoDynamicReferences:
            ref = ctxt.lookup(arg)
            if isinstance(ref, TransientBuffer):
                closureStructArgsType[ctxt._mangle(arg)] = PointerClass(VoidType)
            elif not isinstance(ref, StructBuffer):
                closureStructArgsType[ctxt._mangle(arg)] = ref._type

            if not isinstance(ref, StructBuffer):
                closureStruct[ctxt._mangle(arg)] = arg

        structClass = StructClass(self.closureName + "_args_t", closureStructArgsType)
        self.closureStructArgType = structClass
        self.closureStructArgs = structClass(closureStruct, ctxt)

    # Don't override this
    def _generateClosureCtxt(self, ctxt: NetworkContext, nodeName: str) -> NetworkContext:
        """
        Generate closure context and global definitions.

        Creates the closure function definition and struct type definition,
        then hoists them to the global scope. This includes generating
        the actual closure function code and the argument struct typedef.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context to modify with global definitions.
        nodeName : str
            The name of the node for tracking dependencies.

        Returns
        -------
        NetworkContext
            The modified network context with closure definitions added.

        Notes
        -----
        This method generates and hoists the following global definitions:
        - Closure argument struct typedef
        - Closure function definition with argument casting and optional writeback
        """

        ret = ctxt.hoistStruct(self.closureStructArgs, self.closureName + "_args", self.closureStructArgType)
        ctxt.lookup(ret)._users.append(nodeName)

        allArgs = {
            "closureName": self.closureName,
            "functionCall": self.functionCall,
            "closureStructArgs": ctxt.lookup(self.closureName + "_args").structDict,
            "closureStructArgName": self.closureName + "_args",
            "writeback": self.writeback
        }

        # SCHEREMO: These are global definitions
        closure = self.closureTemplate.generate(allArgs)
        closureStructDef = self.closureStructDefTemplate.generate(allArgs)
        closureStructName = self.closureName + '_args_t'

        ctxt.hoistGlobalDefinition(closureStructName, closureStructDef)
        ctxt.hoistGlobalDefinition(self.closureName, closure)

        return ctxt

    # Don't override this
    def _generateClosureCall(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                             nodeName: str) -> Tuple[NetworkContext, ExecutionBlock]:
        """
        Generate the closure call and replace the original execution block.

        Creates a new ClosureExecutionBlock that wraps the original execution
        with closure call code. This includes the closure function call and
        optional argument writeback code.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context for code generation.
        executionBlock : ExecutionBlock
            The original execution block to wrap with closure calls.
        nodeName : str
            The name of the node for struct generation.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The modified network context
            - The new ClosureExecutionBlock with closure calls

        Notes
        -----
        This method replaces the original function call with:
        1. A closure function call (added to the left)
        2. Optional argument writeback code (added to the right if enabled)
        3. Optional argument struct generation
        """

        allArgs = {
            "closureName": self.closureName,
            "functionCall": self.functionCall,
            "closureStructArgs": ctxt.lookup(self.closureName + "_args").structDict,
            "closureStructArgName": self.closureName + "_args",
            "writeback": self.writeback
        }

        executionBlock = ClosureExecutionBlock(None, executionBlock)

        # SCHEREMO: These replace the function call
        executionBlock.addLeft(self.closureCallTemplate, allArgs)
        if self.writeback:
            executionBlock.addRight(self.closureWriteBackTemplate, allArgs)
        if self.generateStruct:
            ctxt, executionBlock = ArgumentStructGeneration().apply(ctxt, executionBlock, nodeName, _NoVerbosity)

        return ctxt, executionBlock

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """
        Apply the closure generation transformation.

        Transforms the given execution block into a closure-based pattern
        by generating the necessary struct, closure function, and call site.
        This is the main entry point for the closure transformation.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer and type information.
        executionBlock : ExecutionBlock
            The execution block to transform into a closure pattern.
        name : str
            The base name for generating closure-related identifiers.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Default is _NoVerbosity.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The modified network context with closure definitions
            - The new ClosureExecutionBlock with closure call patterns

        Notes
        -----
        The transformation process includes:
        1. Generating a unique closure name with the specified suffix
        2. Capturing the original function call code
        3. Creating the closure argument struct
        4. Generating the closure function definition in global scope
        5. Replacing the original call with a closure call pattern
        """

        # Prepend underscore to avoid name issues when beginning with problematic characters (like numbers)
        self.closureName = "_" + name + self.closureSuffix
        self.functionCall = executionBlock.generate(ctxt)
        self._generateClosureStruct(ctxt, executionBlock)
        ctxt = self._generateClosureCtxt(ctxt, name)
        ctxt, executionBlock = self._generateClosureCall(ctxt, executionBlock, name)
        return ctxt, executionBlock
