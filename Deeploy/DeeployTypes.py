# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
import os
import pickle
import re
import time
from abc import abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Type, TypeVar, Union

import mako
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from mako.template import Template
from onnx.external_data_helper import convert_model_to_external_data
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Logging import FAILURE_MARK, SUCCESS_MARK

from .AbstractDataTypes import BaseType, FloatImmediate, IntegerImmediate, Pointer, PointerClass, Struct, VoidType

Shape = TypeVar("Shape", bound = Any)
SubGraph = List[gs.Node]
Schedule = Union[List[SubGraph], SubGraph]

OperatorRepresentation = Dict[str, Union[
    str,
    Any]]  # Represents the expressions used to describe an operator's parametrization. This is populated by the parser, typechecker, and nodebinding.


@dataclass
class CodeSnippet:
    """A dataclass to hold a NodeTemplate and its associated OperatorRepresentation; used to generate code"""
    template: NodeTemplate
    operatorRepresentation: OperatorRepresentation


@dataclass
class CodeGenVerbosity:
    """
    Encapsulates verbosity options for downstream configuration
    """

    tilingProfiling: Optional[bool] = False  # Specifies if we should profile the tiling code
    untiledProfiling: Optional[bool] = None  #  Specifies if we should profile the untilied code


_NoVerbosity = CodeGenVerbosity(None)

_middlewarePreLoweringFilename = 'middleware_pre_lowering'
_middlewarePostLoweringFilename = 'middleware_post_lowering'
_backendPostParsingFilename = 'backend_post_parsing'
_backendPostBindingFilename = 'backend_post_binding'

_ctxtExtension = '.pkl'
_graphExtension = '.onnx'
_dataExtension = '.data'


# SCHEREMO: mako.Templates are not copiable, since they can use shared context.
# In Deeploy we only use them by direct call (no shared context), so we can override deepcopy and workaround the issue
class _Template(Template):
    """
    This class wraps the Mako.Template class in a way that enables deep-copying
    """

    def __deepcopy__(self, memo):
        _copy = type(self)("", strict_undefined = self.strict_undefined)
        _copy._source = self._source
        _copy._code = self._code
        _copy.module = self.module
        _copy.callable_ = self.callable_
        memo[id(self)] = _copy
        return _copy


class NodeTemplate():
    """This class wraps a `Mako.Template` with additional functionality for hoisting transient buffers and adding expressions to the parsers' node representation"""

    def __init__(self, templateStr: str):
        """Initialize a NodeTemplate object

        Parameters
        ----------
        templateStr : str
            Mako template string. If tiling is supposed to be
            supported, this template string may only contain direct
            expressions that get added by either the operator's parser
            or the `alignToContext` method.

        """
        self.template = _Template(templateStr, strict_undefined = True)
        self.subTemplates: Dict[str, Tuple[NodeTemplate, Callable[[NetworkContext, OperatorRepresentation],
                                                                  Tuple[NetworkContext, OperatorRepresentation]]]] = {}
        self.subTemplateGenerators = {}

    def internalSize(self) -> int:
        """Return the byte size of internal memory buffers used by this template

        Returns
        -------
        int
            byte size of all transient internal buffers

        """
        return 0

    # Override this. Used to hoist optional structs, constants and so on to the GLOBAL context for specialized kernels
    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        """Helper method to extract Mako template expressions used in the backend's code generation step. Also hoists transient buffers.

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext. Modifying is allowed within this method.
        operatorRepresentation : OperatorRepresentation
            Current node representation. Modifying is allowed within this method.

        Returns
        -------
        Tuple[NetworkContext, OperatorRepresentation, List[str]]
            Tuple of the updated NetworkContext, operatorRepresentation and a list of
            the names of hoisted transient buffers


        """
        return ctxt, operatorRepresentation, []

    # Override this
    def computeTransientBuffersSize(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        """Computes the size of transient buffers hoisted by this template given expressions for each variable added by the operator's parser.

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        operatorRepresentation : OperatorRepresentation
            The parser's node representation

        Returns
        -------
        List[Tuple[str, Union[int, IntVar]]]
            Returns a list of tuples containing the hoisted buffer's
            name and either a symbolic expression or an integer
            representing its size.

        """
        return []

    # Override this
    def hoistTransientBuffers(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        """Registers the transient buffers required by this template. If tiling is applied, this method is called AFTER tiling.

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        operatorRepresentation : OperatorRepresentation
            The parser's node representation

        Returns
        -------
        Tuple[NetworkContext, OperatorRepresentation, List[str]]
            Tuple containing the updated `NetworkContext` object,
            updated node representation and a list of names of all
            hoisted `TransientBuffers`

        """
        return ctxt, operatorRepresentation, []

    # Don't override this
    def _alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        ctxt, operatorRepresentation, nameList = self.alignToContext(ctxt, operatorRepresentation)
        for key, (template, repGenerator) in self.subTemplates.items():
            ctxt, subNodeRep, _nameList = template.alignToContext(*(repGenerator(ctxt, operatorRepresentation.copy())))
            self.subTemplateGenerators[key] = (template, copy.copy(subNodeRep))
            nameList += _nameList
        return ctxt, operatorRepresentation, nameList

    # Don't override this
    def generate(self, operatorRepresentation = {}, **kwargs) -> str:
        """Generated the operator's C implementation

        Parameters
        ----------
        operatorRepresentation : The parser's node representation

        Returns
        -------
        str
            Returns the operator's C implementation

        Raises
        ------
        KeyError
            Raises an error whenever an expression in the
            `NodeTemplate`'s templateString is not matched against the
            available expressions in the `operatorRepresentation`

        """
        callStack = ""

        try:
            for key, (template, subNodeRep) in self.subTemplateGenerators.items():
                operatorRepresentation[f'RENDER_{key}'] = template.generate(**subNodeRep, **kwargs)
            callStack += self.template.render(**operatorRepresentation, **kwargs)
        except:
            log.error(operatorRepresentation)
            log.error(mako.exceptions.text_error_template().render())
            raise KeyError(f"Template {self} failed!")
        return callStack


class VariableBuffer():
    """This class represents memory locations containing variable tensor data that is not transient, i.e. intermediate results or input- and output buffers.

    """

    initTemplate: NodeTemplate  #: NodeTemplate: Holds the buffer's initialization code
    allocTemplate: NodeTemplate  #: NodeTemplate: Holds the buffer's allocation code
    deallocTemplate: NodeTemplate  #: NodeTemplate: Holds the buffer's deallocation code

    def __init__(self, name: str = '', shape = [1], aliases: Optional[List[str]] = None):
        self.name: str = name  #: str: Canonical name that this buffer is registered as in the NetworkContext
        self.shape: Sequence[
            int] = shape  #: Sequence[int]: Represents the dimensions of the underlying tensor as a sequence of dimension sizes

        self._users: List[gs.Node] = [
        ]  #: List[gs.Node]: DO NOT OVERRIDE - this variable stores all downstream users of this buffer
        self._type: Type[
            Pointer]  #: Type[Pointer]: DO NOT OVERRIDE - this variable stores the type assigned by the type checking pass
        self._instance: Pointer  #: Pointer: DO NOT OVERRIDE - this variable stores an instantiated POinter assigned by the type checking pass
        self._live: bool = False  #: bool: DO NOT OVERRIDE - this variable is true if a previous Memory allocation pass has allocated the buffer, and false if this buffer has been deallocated or has not been allocated yet.
        self._deploy: bool = True  #: bool: MAY OVERRIDE - this variable is a global switch to deactivate the buffer for all purposes without deleting it outright.

        self._signed = None
        self.nLevels = None

        self.is_input: bool = False
        self.is_output: bool = False

        self.aliases: Set[str] = set(aliases) if aliases is not None else set()

    def _bufferRepresentation(self) -> Dict:
        return {"type": self._instance, "name": self.name, "size": int(np.prod(self.shape))}

    def init(self) -> str:
        """Return a string representation of the C code to declare this memory buffer

        Returns
        -------
        str
            C Code to declare this buffer

        """
        return self.initTemplate.generate(self._bufferRepresentation())

    def alloc(self) -> str:
        """Return a string representation of the C code required to allocated this memory buffer

        Returns
        -------
        str
            C Code to allocate this buffer


        """

        return self.allocTemplate.generate(self._bufferRepresentation())

    def dealloc(self) -> str:
        """Return a string representation of the C code to deallocate/free this memory buffer at runtime

        Returns
        -------
        str
            C Code to free this buffer

        """
        return self.deallocTemplate.generate(self._bufferRepresentation())

    def __str__(self) -> str:
        if hasattr(self, "_type"):
            return f'VariableBuffer: name: {self.name}, type: {self._type}'

        return f'VariableBuffer: name: {self.name}'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        ret = all([self.name == other.name, self.shape == other.shape])
        return ret

    def __getstate__(self):
        d = dict(self.__dict__)
        if 'allocTemplate' in d.keys():
            del d['allocTemplate']
        if 'deallocTemplate' in d.keys():
            del d['deallocTemplate']
        if 'initTemplate' in d.keys():
            del d['initTemplate']
        return d

    @classmethod
    def fromNode(cls, node: gs.Node):
        return (cls(name = node.name, shape = node.shape if not isinstance(node, gs.Constant) else node.values.shape))

    def has_live_aliases(self, ctxt: NetworkContext) -> bool:
        """Checks whether this VariableBuffer has any live aliases, i.e. buffers that are still live and are aliased by this buffer.
        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        Returns
        -------
        bool
            True if this VariableBuffer has any live aliases, False otherwise
        """
        # Do a breadth-first search across the aliasing double-linked list
        live = self._live
        queue = set(self.aliases)
        visited = set(self.name)
        while len(queue) > 0:
            next = queue.pop()
            buffNext = ctxt.lookup(next)
            assert isinstance(buffNext, VariableBuffer)
            live |= buffNext._live
            visited.add(next)
            queue |= buffNext.aliases - visited
        return live

    @property
    def sizeInBytes(self) -> int:
        """Returns the size of this VariableBuffer in bytes

        Returns
        -------
        int
            Size of this VariableBuffer in bytes

        """
        return (math.prod(self.shape) * (self._type.referencedType.typeWidth)) // 8


class TransientBuffer(VariableBuffer):
    """Class to represent memory space required by kernels that is not covered by input and output tensors, e.g. im2col buffers in convolutions


    """

    def __init__(self, name: str = '', size = 0):
        super().__init__(name, shape = (size,))

        # Do not override - Should be written in the parsing passes
        self._type: Type[Pointer] = PointerClass(VoidType)
        self.size = size

    def __eq__(self, other):
        ret = all([self.name == other.name, self.size == other.size])
        return ret

    def _bufferRepresentation(self) -> Dict:
        return {"type": self._type, "name": self.name, "size": int(self.size)}

    def __str__(self) -> str:
        return f'TransientBuffer: name: {self.name}, size: {self.size}'

    def __repr__(self) -> str:
        return f'TransientBuffer: name: {self.name}, size: {self.size}'

    @classmethod
    def fromVariableBuffer(cls, buffer: VariableBuffer):
        ret = cls(name = buffer.name, size = buffer.sizeInBytes)

    @property
    def sizeInBytes(self) -> int:
        return int(self.size)


class ConstantBuffer(VariableBuffer):
    """Class to represent compile-time constant tensors (weights, biases, other parameters) within Deeploy.

    """

    def __init__(self, name: str = '', shape = [1], values = [0]):
        super().__init__(name, shape)
        values = np.asarray(values)
        # intArray = values.astype(int)
        # assert (np.abs(values - intArray)).max() < 0.001, "Constant value {name} is NOT an integer!"
        self.values = values  #: np.array: Stores the underlying weights in Python-type representation

        # Do not override - ConstantBuffers are assumed to be always live!
        self._live = True

    def __eq__(self, other):
        ret = all([super().__eq__(other), np.array_equal(self.values, other.values)])
        return ret

    def _valueString(self) -> str:
        values = list(self.values.reshape(-1))
        if self._type.typeName == 'float32_t*':
            strValues = [f'{value}f' for value in values]
        elif self._type.typeName == 'int8_t*':
            strValues = [f'{int(value)}' for value in values]
        else:
            strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)
        return valueString

    def __str__(self) -> str:
        return f'ConstantBuffer: name: {self.name}, type: {self._type}'

    def __repr__(self) -> str:
        return f'ConstantBuffer: name: {self.name}, type: {self._type}'

    def _bufferRepresentation(self) -> Dict:
        return {"type": self._type, "name": self.name, "size": int(np.prod(self.shape)), "values": self._valueString()}


class StructBuffer(VariableBuffer):
    """Class to represent Struct object needed by the generated C Code

    """

    def __init__(self, name: str, structDict: Dict):
        super().__init__(name, None)
        self.structDict = structDict

    def __eq__(self, other):
        ret = super().__eq__(other) and hasattr(other, "structDict") and self.structDict == other.structDict
        return ret

    def __str__(self) -> str:
        return f'StructBuffer: name: {self.name}, type: {self._type}'

    def __repr__(self) -> str:
        return f'StructBuffer: name: {self.name}, type: {self._type}'

    def _bufferRepresentation(self) -> Dict:
        return {"type": self._type, "name": self.name, "size": int(self._type.typeWidth), "structDict": self.structDict}


class GlobalDefinition():
    """Helper class to hoist arbitrary C code into the global program scope; used to perform small amounts of global initialization, declare global synchronization objects, and similar.

    """

    def __init__(self, name: str, definition: str):
        self.name = name
        self.definition = definition

    def alloc(self) -> str:
        """Return this GlobalDefintion's C code
        """
        return self.definition

    def __eq__(self, other):
        ret = all([self.name == other.name, self.definition == other.definition])
        return ret


class _ReferenceBuffer(VariableBuffer):
    """Helper class to hoist references to pre-established pointers; this is used most frequently in tiling to express an offset with respect to input or output tensors
    """

    allocTemplate = NodeTemplate("""\\
    % if offset is None:
    ${type.typeName} ${name} = (${type.typeName}) ${referenceName};\\
    % else:
    ${type.typeName} ${name} = (${type.typeName})((char*) ${referenceName} + ${offset});\\
    % endif
    """)
    deallocTemplate = NodeTemplate("")
    initTemplate = NodeTemplate("")

    def __init__(self,
                 name: str,
                 reference: VariableBuffer,
                 shape: Tuple[int, ...] = (1,),
                 offset: Optional[Union[int, str, VariableBuffer]] = None):
        super().__init__(name, shape)
        self._referenceName = reference.name
        if isinstance(offset, VariableBuffer):
            self._offset = offset.name
        else:
            self._offset = offset

    def _bufferRepresentation(self) -> Dict:
        repr = super()._bufferRepresentation()
        repr['referenceName'] = self._referenceName
        repr['offset'] = self._offset
        return repr

    def __str__(self) -> str:
        if hasattr(self, "_type"):
            return f'VariableBuffer: name: {self.name}, type: {self._type}, reference: {self._referenceName}+{self._offset}'

        return f'VariableBuffer: name: {self.name}, reference: {self._referenceName}+{self._offset}'

    def __repr__(self) -> str:
        return self.__str__()


class NetworkContext():
    """The global context of the compiler. This object holds all the typing inferred in the type-checking passes within
    the respective buffers. It holds all hoisted transient buffers, struct buffers, and global definitions.
    The context is the source of truth for all code generation in the backend.
    """

    def __init__(self,
                 variableBuffer: Type[VariableBuffer],
                 constantBuffer: Type[ConstantBuffer],
                 structBuffer: Type[StructBuffer],
                 transientBuffer: Type[TransientBuffer],
                 globalObjects: Optional[OrderedDict] = None,
                 localObjects: Optional[OrderedDict] = None,
                 name: str = 'DeeployNetwork'):
        self.globalObjects = globalObjects if globalObjects is not None else OrderedDict()
        self.localObjects = localObjects if localObjects is not None else OrderedDict()
        self.VariableBuffer = variableBuffer
        self.ConstantBuffer = constantBuffer
        self.StructBuffer = structBuffer
        self.TransientBuffer = transientBuffer
        self.name = name

        self._maxDynamicSize = {}  #: int: Maximum dynamic memory size occupied by live buffers at any point in time
        self._dynamicSize = {}  #: int: Current dynamic memory size occupied by live buffers

    def dealiasBuffer(self, name: str) -> str:
        """Function to find the underlying aliased VariableBuffer

        Parameters
        ----------
        name: str
            Name of the VariableBuffer to dealias

        Returns
        -------
        str
            Name of the original VariableBuffer that was referenced

        Raises
        ------
        Exception
            Raises an Exception if aliases are circular, i.e. there
            is no underlying VariableBuffer
        """
        seenAliases: Set[str] = set()
        alias = self.lookup(name)
        while hasattr(alias, "_alias"):
            seenAliases.add(alias.name)
            alias = self.lookup(alias._alias)
            assert alias.name not in seenAliases, "Circular aliasing detected!"
        return alias.name

    def unravelReference(self, ref: VariableBuffer) -> VariableBuffer:
        """Function to find the underlying referenced VariableBuffer

        Parameters
        ----------
        ref : VariableBuffer
            Buffer to unravel

        Returns
        -------
        str
            Name of the original VariableBuffer that was referenced

        Raises
        ------
        Exception
            Raises an Exception if references are circular, i.e. there
            is no underlying VariableBuffer
        """
        seenRefs = set()
        while isinstance(ref, _ReferenceBuffer):
            seenRefs.add(ref.name)
            ref = self.lookup(ref._referenceName)
            assert ref.name not in seenRefs, "Circular reference found"
        return ref

    def exportNetworkContext(self, folderPath: str, fileName: str):
        """Exports the NetworkContext as a pickled dictionary

        Parameters
        ----------
        folderPath : str
            Path to the location where this pickled context should be
            saved
        fileName : str
            Name of the pickled context file

        Raises
        ------
        OSError
            Raises an OSError if the path is not valid

        """
        relativePath = os.path.join(folderPath, fileName + _ctxtExtension)
        absolutePath = os.path.abspath(relativePath)

        if not os.path.isabs(absolutePath):
            raise OSError(f"Error exporting the context to: {absolutePath}")

        with open(absolutePath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def importNetworkContext(folderPath, fileName):
        """Imports a pickled NetworkContext that was saved using exportNetworkContext

        Parameters
        ----------
        folderPath : str
            Path to the location where the pickled context is stored
        fileName : str
            Name of the pickled context file

        Raises
        ------
        OSError
            Raises in OSError if the pickled context file does not
            exist


        """
        relativePath = os.path.join(folderPath, fileName + _ctxtExtension)
        absolutePath = os.path.abspath(relativePath)

        if not os.path.isabs(absolutePath) or not os.path.exists(absolutePath):
            raise OSError(f"File or path does not exist: {absolutePath}")

        with open(absolutePath, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        globalObjects = []
        localObjects = []
        for item in self.globalObjects.values():
            globalObjects.append(str(item))
        for item in self.localObjects.values():
            localObjects.append(str(item))
        _repr = "globalObjects: {\n"
        _repr += ",\n ".join(globalObjects)
        _repr += "} \n\n"
        _repr += "localObjects: {\n"
        _repr += ",\n ".join(localObjects)
        _repr += "}"
        return _repr

    def __eq__(self, other):
        if not isinstance(other, NetworkContext):
            raise TypeError(f'Cannot compare NetworkContext with {type(other)}!')

        if not other.globalObjects.keys() == self.globalObjects.keys():
            return False

        if not other.localObjects.keys() == self.localObjects.keys():
            return False

        for buffer_name in self.globalObjects.keys():
            if not self.globalObjects[buffer_name] == other.globalObjects[buffer_name]:
                return False

        for buffer_name in self.localObjects.keys():
            if not self.localObjects[buffer_name] == other.localObjects[buffer_name]:
                return False

        return True

    def _mangle(self, name: str, repr: bool = True) -> str:
        repStr = name
        repStr = re.sub('\.', '_', repStr)
        repStr = re.sub(':', '_', repStr)
        if repr:
            repStr = re.sub('\.', '_', self.name) + '_' + repStr
        return repStr

    def add(self, obj: VariableBuffer, ctxt: Literal['local', 'global'] = 'local', _id: str = ""):
        """Adds a VariableBuffer object to the NetworkContext

        Parameters
        ----------
        obj : VariableBuffer
            The VariableBuffer object to be registered
        ctxt : str
            Level of the NetworkContext to register the VariableBuffer in, either local or global
        _id : str
            Override for the registration name of the
            VariableBuffer. Do not use unless you have a good reason!

        Raises
        ------
        ValueError
            Raises a ValueError if ctxt is not local or global
        KeyError
            Raises a KeyError if the VariableBuffer's name is already
            registered within the NetworkContext


        """
        if _id != "":
            obj.name = self._mangle(_id + "_" + obj.name, False)

        if ctxt == 'local':
            if obj.name not in self.localObjects.keys():
                self.localObjects[obj.name] = obj
            else:
                raise KeyError(f'Buffername {obj.name} was already in the local context!')
        elif ctxt == 'global':
            if obj.name not in self.globalObjects.keys():
                self.globalObjects[obj.name] = obj
            else:
                raise KeyError(f'Buffername {obj.name} was already in the global context!')
        else:
            raise ValueError("Expected either local or global context")

    def lookup(self, name: str, _id: str = "") -> Union[VariableBuffer, GlobalDefinition]:
        """Returns the VariableBuffer or GlobalDefinition registered under a given name

        Parameters
        ----------
        name : str
            Name of the VariableBuffer to look up
        _id : str
            Override for the registration name of the
            VariableBuffer. Do not use unless you have a good reason!

        Returns
        -------
        Union[VariableBuffer, GlobalDefinition]
            Registered buffer object

        Raises
        ------
        KeyError
            Raises a KeyError if the name does not match with any
            registered object

        """

        if _id != "":
            name = self._mangle(_id + "_" + name, False)

        if name in self.localObjects.keys():
            return self.localObjects[name]
        elif name in self.globalObjects.keys():
            return self.globalObjects[name]
        else:
            raise KeyError(f'Expected key {name} to be in either local or global context!')

    def is_global(self, name: str) -> bool:
        """Checks whether a name is associated with a global buffer

        Parameters
        ----------
        name : str
            Name of the VariableBuffer to check for

        Returns
        -------
        bool
            Returns true if the name matches with any global buffer

        """
        return name in self.globalObjects

    def is_local(self, name: str) -> bool:
        """Checks whether a name is associated with a local buffer

        Parameters
        ----------
        name : str
            Name of the VariableBuffer to check for

        Returns
        -------
        bool
            Returns ture if the name matches with any local buffer

        """
        return name in self.localObjects

    def is_object(self, value: Any) -> bool:
        """Checks whether a value is an existing object name

        Parameters
        ----------
        value : Any
            Value to check

        Returns
        -------
        bool
            Returns ture if the value is an existing buffer name

        """
        return isinstance(value, str) and (self.is_local(value) or self.is_global(value))

    def is_buffer(self, value: Any) -> bool:
        """Checks whether a value is an existing buffer name

        Parameters
        ----------
        value : Any
            Value to check

        Returns
        -------
        bool
            Returns ture if the value is an existing buffer name

        """
        if not self.is_object(value):
            return False
        obj = self.lookup(value)
        return isinstance(obj, VariableBuffer)

    def hoistTransientBuffer(self, name: str, size: int) -> str:
        """Registers a new TransientBuffer in the local context

        Parameters
        ----------
        name : str
            Name of the TransientBuffer to register
        size : int
            BYTE size of the TransientBuffer to register

        Returns
        -------
        str
            On success, return the name of the registered buffer

        """
        transientBuffer = self.TransientBuffer(name, size)
        self.add(transientBuffer, 'local')

        return name

    def hoistGlobalDefinition(self, name: str, definition: str) -> None:
        """Registers a new GlobalDefinition in the global context

        Parameters
        ----------
        name : str
            Name of the GlobalDefinition to register
        definition : str
            Program code of the GlobalDefinition

        """

        _definition = GlobalDefinition(name, definition)
        self.add(_definition, 'global')

    def hoistStruct(self, _struct: Union[Dict[str, BaseType], Struct], name: str, _type: Type[Struct]) -> str:
        """Register a Struct with the local context

        Parameters
        ----------
        _struct : Union[Dict[str, BaseType], Struct]
            Struct object or Struct object's definition
        name : str
            Name to register the struct under
        _type : Type[Struct]
            Type definition of the Struct class to register

        Returns
        -------
        str
            On success, return the name of the registered buffer

        """

        if isinstance(_struct, _type):
            struct = _struct
        else:
            struct = _type(_struct, self)

        structBuffer = self.StructBuffer(name, struct)
        structBuffer._type = _type
        structBuffer._instance = struct
        self.add(structBuffer, 'local')

        return name

    def hoistConstantAndReference(self, constBuf: ConstantBuffer, pointerType: Type[Pointer]) -> str:
        """Helper function to hoist a new ConstantBuffer and a _ReferenceBuffer to it. Mostly used in tiling to create boilerplate for tiled variables.

        Parameters
        ----------
        constBuf : ConstantBuffer
            ConstantBuffer to hoist
        pointerType : Type[Pointer]
            Pointer class to assign to the constant buffer

        Returns
        -------
        str
            name of the registered _ReferenceBuffer

        """
        constBuf._type = pointerType
        self.add(constBuf, "global")
        constBuf._instance = constBuf._type(constBuf.name, self)
        ref = self.hoistReference(constBuf.name + "_ref", constBuf)
        return ref.name

    def hoistReference(self,
                       name: str,
                       reference: VariableBuffer,
                       shape: Tuple[int, ...] = (1,),
                       offset: Union[int, str, VariableBuffer] = 0,
                       override_type: Optional[Type[BaseType]] = None) -> _ReferenceBuffer:
        """Helper function to register a _ReferenceBuffer to a preexisting VariableBuffer

        Parameters
        ----------
        name : str
            Name of the _ReferenceBuffer to register
        reference : VariableBuffer
            Referenced VariableBuffer
        shape: Tuple[int, ...]
            Shape of the reference
        offset: Union[int, str, VariableBuffer]
            Offset from the reference
        override_type: Optional[Type[BaseType]]
            Optional argument to override the reference type

        Returns
        -------
        _ReferenceBuffer
            Returns the newly registered _ReferenceBuffer
        """
        ref = _ReferenceBuffer(name, reference, shape, offset)
        if override_type is not None:
            ref._type = PointerClass(override_type)
        else:
            ref._type = reference._type
        self.add(ref, 'local')
        ref._instance = ref._type(name, ctxt = self)
        return ref

    def hoistConstant(self,
                      constant: gs.Constant,
                      name: Optional[str] = None,
                      _type: Optional[Type[Pointer]] = None) -> str:
        """Register a ConstantBuffer extracted directly from a graphsurgeon Constant

        Parameters
        ----------
        constant : gs.Constant
            graphsurgeon.Node containing a single constant output
        name : str
            Name of the ConstantBuffer to be registered
        _type : Optional[Type[Pointer]]
            Optional type assignment of the registered ConstantBuffer

        Returns
        -------
        str
            Returns the name of the newly registed ConstantBuffer

        """
        assert len(constant.outputs) <= 1, f"Constant {constant.name} has more than one output"

        name = name if name is not None else constant.name

        # LMACAN: The shape needs to be copied into a tuple for pickling to work. Don't ask me why..
        buffer = self.ConstantBuffer(name, tuple(constant.shape), constant.values)
        self.add(buffer, 'global')

        if _type is not None:
            self.annotateType(name, _type)

        return name

    def addUser(self, name: str, node: gs.Node):
        """Adds an operator's name to the _user list of a VariableBuffer in the context

        Parameters
        ----------
        name : str
            Name of the VariableBuffer that gets used by the node
        node : gs.Node
            graphsurgeon Node of the operator

        """

        _buffer = self.lookup(name)
        if node.name not in _buffer._users:
            _buffer._users.append(node.name)
        if self.is_local(_buffer.name):
            self.localObjects[_buffer.name] = _buffer
        else:
            self.globalObjects[_buffer.name] = _buffer

    def annotateType(self, name: str, _type: Type[Pointer]):
        """Annotates a Deeploy-type pointer on the _type field of a VariableBuffer

        Parameters
        ----------
        name : str
            Name of the VariableBuffer to annotate
        _type : Type[Pointer]
            Type of the Deeploy-type pointer to annotate the
            VariableBuffer with

        """
        obj = self.lookup(name)
        obj._type = _type
        obj._instance = _type(name, ctxt = self)

    def copy(self) -> NetworkContext:
        """Return a shallow copy of this NetworkContext

        """
        return copy.copy(self)


class NodeParser():
    """Deeploy's core Parser class. Analyzes network nodes and evaluates whether they can be mapped by it.

    """

    def __init__(self):
        self.operatorRepresentation: OperatorRepresentation = {
        }  #: Dict[str, Any]: The internal representation of the operator this parser has analyzed that describes all relevant attributes of the node to be used by code generation

    @abstractmethod
    def parseNode(self, node: gs.Node) -> bool:
        """Parser-specific method to-be-implemented. Given a graphsurgeon node, this method returns whether its attributes are mappable by this parser.

        Parameters
        ----------
        node : gs.Node
            Graphsurgeon node to be evaluated

        Returns
        -------
        bool
            False if any attribute in the node cannot be mapped
            correctly.

        """
        return True

    @abstractmethod
    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        """Parses the node's input and output tensors, and adds them to its operatorRepresentation. May also be used to assert certain input- and output-level characteristics like correct dimensions.

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Node to be analyzed
        channels_first : bool
            Flag to indicate whether tensor dimensions are expected to
            be in CxHxW layout (true) or HxWxC layout (false)

        Returns
        -------
        Tuple[NetworkContext, bool]
            Tuple of the updated NetworkContext and return boolean to
            indicate whether the node, including it's IO tensors can
            be mapped.

        """

        return ctxt, True

    @classmethod
    def parseInputs(cls, ctxt: NetworkContext, node: gs.Node) -> NetworkContext:
        """DONT OVERRIDE - Takes care of hoisting IO tensors into the NetworkContext. Also verifies
        that all inputs have been registered and the output has not been registered.

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Node whose IO tensors should be hoisted

        Returns
        -------
        NetworkContext
            Updated NetworkContext with hoisted IO tensors

        """
        data_in_buffers = []
        for inputNode in node.inputs:
            data_in = inputNode.name

            # Hoist constant inputs
            if type(inputNode) == gs.ir.tensor.Constant and not ctxt.is_global(data_in):
                ctxt.hoistConstant(inputNode)
            else:
                localBuffer = ctxt.lookup(data_in)
                data_in_buffers.append(localBuffer.name)

            ctxt.addUser(data_in, node)

        return ctxt

    @classmethod
    def parseOutputs(cls, ctxt: NetworkContext, node: gs.Node) -> NetworkContext:
        """DONT OVERRIDE - registers the output tensor of the operator

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Operator whose outputs should be parsed

        Returns
        -------
        NetworkContext
            Updated NetworkContext

        """
        outputNodes = node.outputs
        outputNames = [node.name for node in outputNodes]

        for node, name in zip(outputNodes, outputNames):
            if not ctxt.is_global(name):
                nb = ctxt.VariableBuffer(name = name, shape = node.shape)
                ctxt.add(nb, 'local')
            else:
                nb = ctxt.lookup(name)

        return ctxt

    @staticmethod
    def _unpack_const(attr) -> Union[int, float]:
        """DON'T OVERRIDE - Helper function to get a Python scalar from an ONNX attribute.
        The attributes can either be a numpy scalar value or a Constant tensor.
        This expects the numpy value to be of size 1.
        """
        if isinstance(attr, gs.Constant):
            value = attr.values
        elif isinstance(attr, np.ndarray):
            value = attr
        else:
            assert False, f"Unsupported attribute type {type(attr)}"
        assert value.size == 1, f"Expected attribute of size 1. Got an array of shape {value.shape}"
        return value.item()

    # Don't touch this
    def parse(self,
              ctxt: NetworkContext,
              node: gs.Node,
              default_channels_first: bool = True,
              ioParse: bool = True) -> Tuple[NetworkContext, bool]:
        """DONT OVERRIDE - Uses other NodeParser functions to implement a full parsing passing of the node

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Node to be parsed
        default_channels_first : bool
            The default `channels_first` value if none is provided by the node's attributes
        ioParse : bool
            Flag to indicate whether to go through IO parsing or not

        Returns
        -------
        Tuple[NetworkContext, bool]
            Returns updated NetworkContext and flag to indicate
            success

        """
        self.operatorRepresentation = {}

        if "channels_first" in node.attrs:
            self.operatorRepresentation['channels_first'] = node.attrs['channels_first']
        else:
            self.operatorRepresentation['channels_first'] = default_channels_first

        ret = self.parseNode(node)

        if not ret:
            return ctxt, False

        if ioParse:
            ctxt = ctxt.copy()
            ctxt = self.parseInputs(ctxt, node)
            ctxt = self.parseOutputs(ctxt, node)

        return ctxt, True


class NodeTypeChecker():
    """Implements type checking according to user-defined rules to assign Deeploy-types to the Python-typed input graph

    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        """Generate a type checking rule

        Parameters
        ----------
        input_types : Sequence[Type[Pointer]]
            Ordered sequence of Deeploy-types that should be assigned
            to the operator's Python-typed input tensor
        output_types : Sequence[Type[Pointer]]
            Ordered sequence of Deeploy-types that should be assigned
            to the operator's Python-typed input tensor

        """

        self.input_types = input_types
        self.output_types = output_types

        self.typeDict: Dict[str, Type[Pointer]] = {
        }  #: Dict[str, Type[Pointer]]: Stores the type assignment of the input and output tensors, mapping them to the names defined by the NodeParser

    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        """TypeCheck method to-be-implemented. Returns whether the type checking rule is met or not

        Parameters
        ----------
        inputs : List[VariableBuffer]
            Ordered list of operator inputs to be used for inferring
            the output type
        operatorRepresentation : OperatorRepresentation
            NodeParser's operatorRepresentation

        Returns
        -------
        bool
            True if output type can be assigned as defined in
            output_types

        """
        return True

    def typeInferOutput(self, ctxt: NetworkContext, node: gs.Node,
                        operatorRepresentation: OperatorRepresentation) -> NetworkContext:
        """DONT OVERRIDE - Annotates each VariableBuffer in the NetworkContext corresponding to an output of the operator with this rule's output types.

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Node whose outputs should be annotated
        operatorRepresentation : OperatorRepresentation
            NodeParser's operatorRepresentation

        Returns
        -------
        NetworkContext
            Updated NetworkContext

        """
        newCtxt = ctxt.copy()

        inputs = [ctxt.lookup(inputNode.name) for inputNode in node.inputs]
        outputNames = [node.name for node in node.outputs]

        outputTypes = self.output_types

        for name, output_type in zip(outputNames, outputTypes):
            newCtxt.annotateType(name, output_type)

        return newCtxt

    def typeCheckNodeInputs(self, ctxt: NetworkContext, node: gs.Node) -> bool:
        """DONT OVERRIDE - Type checks all input nodes to confirm they either already are assigned the correct type or their type can be statically upcast to the rule's input types

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Node whose inputs should be analyzes

        Returns
        -------
        bool
            Whether the input's types match the rule's requirements

        """
        retCheck = True

        for inputNode, _type in zip(node.inputs, self.input_types):
            reference = ctxt.lookup(inputNode.name)

            if not isinstance(reference, VariableBuffer):
                return False

            if hasattr(reference, "values"):
                retCheck &= _type.referencedType.checkPromotion(reference.values)
            else:
                if ctxt.is_global(inputNode.name):
                    retCheck &= _type.referencedType.partialOrderUpcast(reference._type.referencedType)

                    if retCheck:
                        reference._type = _type
                        reference._instance = _type(inputNode.name, ctxt)
                else:
                    retCheck &= reference._type.referencedType == _type.referencedType
        return retCheck

    def typeInferGlobalCtxt(self, ctxt: NetworkContext, node: gs.Node) -> NetworkContext:
        for inputNode, _type in zip(node.inputs, self.input_types):
            if isinstance(ctxt.lookup(inputNode.name), ConstantBuffer):
                reference = ctxt.lookup(inputNode.name)
                if not _type.referencedType.checkPromotion(reference.values):
                    raise Exception(f"Can't cast {reference} to {_type}!")

                ctxt.annotateType(inputNode.name, _type)

        return ctxt

    def annotateDict(self, ctxt: NetworkContext, node: gs.Node, operatorRepresentation: OperatorRepresentation):
        """Store the inferred typing information into the rule's type dict

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Operator whose inputs and outputs should be considered
        operatorRepresentation : OperatorRepresentation
            The NodeParser's operatorRepresentation

        """
        env = [node.name for node in node.inputs + node.outputs]
        for key, value in operatorRepresentation.items():
            # check if the referenced buffer is in the environment
            if isinstance(value, str) and value in env:
                self.typeDict[key + '_type'] = ctxt.lookup(value)._type

    def typeCheck(self, ctxt: NetworkContext, node: gs.Node,
                  operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, bool]:
        """DONT OVERRIDE - Uses other NodeTypeChecker methods to implement full type inference on a single node

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Node that should be used for type inference
        operatorRepresentation : OperatorRepresentation
            The NodeParser's operatorRepresentation

        Returns
        -------
        Tuple[NetworkContext, bool]
            Updated NetworkContext and whether type inference was
            successful with this rule.

        """
        newCtxt = ctxt.copy()

        if not self.typeCheckNodeInputs(newCtxt, node):
            return ctxt, False

        if not self.checkOutputType(node.inputs, operatorRepresentation):
            return ctxt, False

        newCtxt = self.typeInferGlobalCtxt(newCtxt, node)
        newCtxt = self.typeInferOutput(newCtxt, node, operatorRepresentation)
        self.annotateDict(newCtxt, node, operatorRepresentation)
        return (newCtxt, True)

    def signature(self) -> str:
        input_types_str = ", ".join([_type.referencedType.typeName for _type in self.input_types])
        output_types_str = ", ".join([_type.referencedType.typeName for _type in self.output_types])
        return f"({input_types_str}) -> {output_types_str}"


class ExecutionBlock():
    """Deeploy abstraction to represent a operator whose kernel has been determined. Mostly used to apply various code transformations, and, finally, generate C Code

    """

    def __init__(self, operatorCodeSnippet: Optional[CodeSnippet] = None):
        """Initialize a new ExecutionBlock object from a CodeSnippet

        Parameters
        ----------
        codeSnippet : Optional[CodeSnippet]
            NodeTemplate + operatorRepresentation combination that makes up this
            ExecutionBlock
        """
        if operatorCodeSnippet is not None:
            self.codeSnippets = deque([operatorCodeSnippet])
        else:
            self.codeSnippets = deque(
                []
            )  #: Sequence[CodeSnippet]: ordered list of code snippets that need to be generated to implemented the associated operator

        self.patternMemoryConstraint: Optional = None  #: Optional[PatternMemoryConstraint]: Tiling information of the operator which is annotated in the midend

    def addLeft(self, template: NodeTemplate, operatorRepresentation: OperatorRepresentation):
        """Adds a code snippet that is generated BEFORE any of the other code snippets in this ExecutionBlock

        Parameters
        ----------
        template : NodeTemplate
            NodeTemplate that represents the code snippet to be added
        operatorRepresentation : OperatorRepresentation
            Dictionary that holds all expressions to generate code
            from the template


        """
        self.codeSnippets.appendleft(CodeSnippet(template, operatorRepresentation))

    def addRight(self, template: NodeTemplate, operatorRepresentation: OperatorRepresentation):
        """Adds a code snippet that is generated AFTER any of the other code snippets in this ExecutionBlock

        Parameters
        ----------
        template : NodeTemplate
            NodeTemplate that represents the code snippet to be added
        operatorRepresentation : OperatorRepresentation
            Dictionary that holds all expressions to generate code
            from the template

        """

        self.codeSnippets.append(CodeSnippet(template, operatorRepresentation))

    def hoisting(self, ctxt: NetworkContext, **kwargs) -> Tuple[NetworkContext, List[str]]:
        """Helper function to run the underlying NodeTemplate's hooks to add TransientBuffers into the NetworkContext and call their alignToContext methods

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext

        Returns
        -------
        Tuple[NetworkContext, List[str]]
            Updated NetworkContext and a list of newly registered
            buffer names

        """

        transientBuffers = []
        contextBuffers = []

        for idx, codeSnippet in enumerate(self.codeSnippets.copy()):

            template, operatorRepresentation = codeSnippet.template, codeSnippet.operatorRepresentation

            newCtxt, operatorRepresentation, _transientBuffers = template.hoistTransientBuffers(
                ctxt, {
                    **operatorRepresentation,
                    **kwargs
                })
            newCtxt, operatorRepresentation, _contextBuffers = template._alignToContext(
                newCtxt, {
                    **operatorRepresentation,
                    **kwargs
                })

            self.codeSnippets[idx].operatorRepresentation.update(operatorRepresentation)
            transientBuffers += _transientBuffers
            contextBuffers += _contextBuffers

        return newCtxt, transientBuffers + contextBuffers

    @staticmethod
    def _mangleOpRepr(ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> OperatorRepresentation:
        mangledOpRepr = {}

        for key, value in operatorRepresentation.items():
            if ctxt.is_buffer(value):
                mangledOpRepr[key] = ctxt._mangle(value)
            else:
                mangledOpRepr[key] = value

        return mangledOpRepr

    def generate(self, ctxt: NetworkContext, **kwargs) -> str:
        """Generates the code for all registered NodeTemplates and joins it to construct a single snippet

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext

        Returns
        -------
        str
            Code snippet that represent the entire ExecutionBlock

        """

        return ("\n").join([
            codeSnippet.template.generate(
                ExecutionBlock._mangleOpRepr(ctxt, {
                    **codeSnippet.operatorRepresentation,
                    **kwargs
                })) for codeSnippet in self.codeSnippets
        ])


class NodeBinding():
    """Deeploy's class to bind individual NodeTypeChecker objects to NodeTemplate and associate a CodeTransformation.

    """

    def __init__(self, typeChecker: NodeTypeChecker, template: NodeTemplate, codeTransformer: CodeTransformation):
        self._typeChecker = typeChecker  #: NodeTypeChecker: The NodeTypeChecker that verifies the kernel template's signature can be matched to the node
        self.template = template  #: NodeTemplate: The kernel template you want to bind
        self._executionBlock: ExecutionBlock = ExecutionBlock(
        )  #: ExecutionBlock: The executionBlock that will be built from the NodeTemplate
        self._nodeName: str
        self.buffers: List[VariableBuffer] = []
        self.codeTransformer: CodeTransformation = codeTransformer

    def __repr__(self):
        return f"{self.template.__class__.__name__}{self._typeChecker.signature()}"

    @property
    def typeChecker(self):
        """Read-only wrapper around the encapsulated type checker
        """
        return self._typeChecker

    @property
    def executionBlock(self):
        """Read-only wrapper around the encapsulated execution block
        """
        return self._executionBlock

    @property
    def nodeName(self):
        """Read-only wrapper around the encapsulated node's name
        """
        return self._nodeName

    def earlyBinding(self, ctxt: NetworkContext, node: gs.Node,
                     operatorRepresentation: OperatorRepresentation) -> NetworkContext:
        """Initializes the executionBlock with the NodeTemplate

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            The operator this NodeBinding is associated with
        operatorRepresentation : OperatorRepresentation
            The NodeParser's operatorRepresentation

        Returns
        -------
        NetworkContext
            Updated NetworkContext

        """
        self.executionBlock.addLeft(self.template, operatorRepresentation)
        self._nodeName = operatorRepresentation['nodeName']
        return ctxt

    def typeCheck(self, ctxt: NetworkContext, node: gs.Node,
                  operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, bool]:
        """Runs the binding-level typechecker on a node

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            The node to be typechecked
        operatorRepresentation : OperatorRepresentation
            The NodeParser's operatorRepresentation

        Returns
        -------
        Tuple[NetworkContext, bool]
            Updated and NetworkContext and true if the typing rule
            matches the node

        """

        newCtxt, ret = self.typeChecker.typeCheck(ctxt.copy(), node, operatorRepresentation)
        if ret:
            log.debug(f" {SUCCESS_MARK} Type check passed for {self}")
            return newCtxt, True
        else:
            log.debug(f" {FAILURE_MARK} Type check failed for {self}")

        return ctxt, False

    def bind(self, ctxt: NetworkContext, node: gs.Node,
             operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, List[str], bool]:
        """Initializes the executionBlock and hoist all necessary transient buffers of the underlying NodeTemplate

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            The node that should be bound
        operatorRepresentation : OperatorRepresentation
            The NodeParser's operatorRepresentation
        Returns
        -------
        Tuple[NetworkContext, List[str], bool]
            Updated NetworkContext, a list of names of transient
            buffers that were hoisted and true if binding succeeded

        """
        newCtxt = self.earlyBinding(ctxt, node, operatorRepresentation)
        newCtxt, buffers = self.executionBlock.hoisting(newCtxt, **self.typeChecker.typeDict)

        for _buffer in buffers:
            newCtxt.lookup(_buffer)._users.append(self._nodeName)

        return newCtxt, [], True

    def codeTransform(self, ctxt: NetworkContext, verbose: CodeGenVerbosity = _NoVerbosity):
        """Applies the CodeTransformer's passes on the executionBlock

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        verbose : CodeGenVerbosity
            Verbosity options to control code generation

        """
        ctxt, self._executionBlock = self.codeTransformer.transform(ctxt, self.executionBlock, self.nodeName, verbose)
        return ctxt

    def generate(self, ctxt: NetworkContext) -> List[str]:
        """Generates C Code from the encapsulated executionBlock

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext

        Returns
        -------
        List[str]
            A list of C Code snippets to be pasted into the final
            program

        """
        nodeCall = self.executionBlock.generate(ctxt, **self.typeChecker.typeDict)
        return [nodeCall]


class NodeMapper():
    """Deeploy class to link a NodeParser and several NodeBindings
    """

    def __init__(self, parser: NodeParser, bindings: List[NodeBinding]):
        self.parser = parser  #: NodeParser: The NodeParser object which is used to determine whether an operator may be bound to one of the associated bindings
        self.bindings = bindings  #: List[NodeBinding]: All possible bindings that correspond to the linked parser

        self.binder: NodeBinding  #: NodeBinding: The currently chosen NodeBinding
        self.bound = False  #: bool: Indicates whether a binder has been chosen or not

        self.discardedBindings = set()  #: Set[NodeBinding]: Set of all bindings which have been tried unsuccessfully.

    def __repr__(self):
        bindings_str = "\n  ".join([repr(binding) for binding in self.bindings])
        return f"{self.parser.__class__.__name__} [\n  {bindings_str}\n]"

    # Don't override this. Parses the networks with the correct data type
    def _parse(self,
               ctxt: NetworkContext,
               node: gs.Node,
               default_channels_first: bool = True,
               ioParse: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = self.parser.parse(ctxt.copy(), node, default_channels_first, ioParse)
        if ret:
            log.debug(f" {SUCCESS_MARK} Parser {self.parser.__class__.__name__} succeeded")
            return newCtxt, True
        else:
            log.debug(f" {FAILURE_MARK} Parser {self.parser.__class__.__name__} failed")

        return ctxt, False

    def _parseCtxt(self,
                   ctxt: NetworkContext,
                   node: gs.Node,
                   default_channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = self.parser.parseNodeCtxt(ctxt.copy(), node, default_channels_first)
        if ret:
            log.debug(f" {SUCCESS_MARK} Context parsing succeeded with {self.parser.__class__.__name__}")
        else:
            log.debug(f" {FAILURE_MARK} Context parsing failed with {self.parser.__class__.__name__}")
        return (newCtxt, ret)

    def bindingsExhausted(self) -> bool:
        """Returns whether all bindings have been tried

        Returns
        -------
        bool
            True is no more bindings are possible

        """
        return len(self.discardedBindings) == len(self.bindings)

    def discardCurrentBinder(self):
        """Discards the binder object

        """
        self.discardedBindings.add(self.binder)
        self.binder = None
        self.bound = False

    def resetDiscardedBindings(self):
        """Reset the discardedBindings set

        """
        self.discardedBindings = set()

    def typeCheck(self, ctxt: NetworkContext, node: gs.Graph) -> Tuple[NetworkContext, bool]:
        """Tries to elect a binder object whose typeChecker allows the node configuration

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Graph
            The node that is being evaluated

        Returns
        -------
        Tuple[NetworkContext, bool]
            Updated NetworkContext and bool to indicate success or
            failure

        """

        for idx, binder in enumerate(self.bindings):

            if binder in self.discardedBindings:
                continue

            newCtxt, ret = binder.typeCheck(ctxt.copy(), node, self.parser.operatorRepresentation)

            if not ret:
                self.discardedBindings.add(binder)
                continue

            self.binder = binder
            return newCtxt, True

        log.debug(f" ‼ All {len(self.bindings)} bindings exhausted for {self.parser.__class__.__name__}")
        return ctxt, False

    # Don't override this. This should annotate the output node with the correct data type
    # SCHEREMO: Currently simply binds the first viable binding
    def bind(self, ctxt: NetworkContext, node: gs.Node) -> Tuple[NetworkContext, List[str], bool]:
        """Invokes the binder's bind method to setup the executionBlock and buffer hoisting

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        node : gs.Node
            Node that should be bound

        Returns
        -------
        Tuple[NetworkContext, List[str], bool]
            Updated NetworkContext, list of hoisted TransientBuffers'
            names, boolean to indicate success or failure

        """

        newCtxt, transientBuffers, ret = self.binder.bind(ctxt.copy(), node, self.parser.operatorRepresentation)
        if not ret:
            return ctxt, [], False

        self.bound = True

        return newCtxt, transientBuffers, True

    def generate(self, ctxt: NetworkContext) -> List[str]:
        """Generates the C Code of the binder elected by this mapper

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext

        Returns
        -------
        List[str]
            Returns a list of code snippets that correspond to the
            operator's invocation

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if no binder has been elected or the
            binder has not been bound yet.

        """
        if not self.bound:
            raise RuntimeError("Bind layer before generating code!")
        return self.binder.generate(ctxt)


class ONNXLayer():
    """Deeploy abstraction to represent one operator in an ONNX graph
    """

    def __init__(self, maps: List[NodeMapper]):
        self.maps = maps  #: List[NodeMapper]: All potential mappings of an ONNX Layer

        self.mapper: NodeMapper  #: NodeMapper: The currently elected NodeMapper to represent this layer
        self.discardedMappers: Set[NodeMapper] = set(
        )  #: Set[NodeMapper]: Set of all NodeMappers which cannot be used to represent this layer
        self.node: gs.Node = None  #: gs.Node: The represented operator

    def __repr__(self):
        maps_str = "\n  ".join([repr(mapper) for mapper in self.maps])
        return f"{self.__class__.__name__}(maps=[\n  {maps_str}\n])"

    def computeOps(self):
        """Returns the number of operations (1 MAC = 2 Ops) of this operator
        """
        assert self.mapper is not None, "To compute Ops, network must first be parsed!"

        return 0

    # Override this for broadcasting support
    # Returns a tuple of new, broadcasted inputShapes and outputShapes
    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation: OperatorRepresentation,
                      channels_first: bool) -> Tuple[Shape, Shape]:
        """Takes input and output shapes from the graph-representation and broadcasts them to a predefined layout

        Parameters
        ----------
        inputShapes : Shape
            Graph-level input shape
        outputShapes : Shape
            Graph-level output shapes
        operatorRepresentation : OperatorRepresentation
            The node's operatorRepresentation
        channels_first : bool
            Whether this operator's data layout is in CxHxW (true) or
            HxWxC (false) layout

        Returns
        -------
        Tuple[Shape, Shape]
            Returns broadcasted shapes

        """
        return (inputShapes, outputShapes)

    def broadcast(self, ctxt: NetworkContext, default_channels_first: bool = True) -> (NetworkContext):
        """Broadcasts the operator's shapes and updates the NetworkContext

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        default_channels_first : bool
            Whether the default layout if channels-first or not

        Returns
        -------
        (NetworkContext)
            Updated NetworkContext

        Raises
        ------
        KeyError
            Raises a KeyError if any tensor required is not found in
            the NetworkContext
        RuntimeError
            Raises a RuntimeError if any tensor's shape could not be
            broadcast to the target shape

        """
        inputShapes = [ctxt.lookup(node.name).shape for node in self.node.inputs]
        outputShapes = [ctxt.lookup(node.name).shape for node in self.node.outputs]

        if not "channels_first" in self.mapper.parser.operatorRepresentation:
            channels_first = default_channels_first
        else:
            channels_first = self.mapper.parser.operatorRepresentation['channels_first']

        newInputShapes, newOutputShapes = self.computeShapes(inputShapes, outputShapes,
                                                             self.mapper.parser.operatorRepresentation, channels_first)

        for node, newShape in zip(self.node.inputs + self.node.outputs, newInputShapes + newOutputShapes):
            if ctxt.is_local(node.name):
                ctxt.localObjects[node.name].shape = newShape
                # Update shape of tensors in onnx graph
                node.shape = newShape

                # WIESEP: It is possible that the type was not yet set, so we assume some default type
                # At this state, we assume that all local buffers are float32 type inference is not yet done.
                if node.dtype is None:
                    node.dtype = np.float32

            elif ctxt.is_global(node.name):
                ctxt.globalObjects[node.name].shape = newShape
                if isinstance(ctxt.globalObjects[node.name], ConstantBuffer):

                    # If the number of elements is equal, reshape
                    if np.prod(ctxt.globalObjects[node.name].values.shape) == np.prod(newShape):
                        ctxt.globalObjects[node.name].values.reshape(newShape)
                    # The number of elements SHOULD be lower, and we broadcast
                    else:
                        try:
                            ctxt.globalObjects[node.name].values = np.broadcast_to(ctxt.globalObjects[node.name].values,
                                                                                   newShape)
                        except:
                            raise RuntimeError(
                                f"Could not broadcast {node.name} from {ctxt.globalObjects[node.name].values.shape} to {newShape}!"
                            )

            else:
                raise KeyError(f'Expected node {node.name} to be in context!')

        return ctxt

    # Don't override - binds the layer to a node
    def __call__(self, node: gs.Node):
        _copy = copy.deepcopy(self)
        _copy.node = node
        return _copy

    def discardCurrentMapper(self):
        """Discard the current Mapper

        """
        self.discardedMappers.add(self.mapper)
        self.mapper = None

    def resetDiscardedMappers(self):
        """Reset all discarded mappers

        """
        for mapper in self.maps:
            mapper.resetDiscardedBindings()
        self.discardedMappers = set()

    def parse(self, ctxt: NetworkContext, default_channels_first: bool) -> Tuple[NetworkContext, bool]:
        """Iterate through all possible mappers and elect the first one that work

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        default_channels_first : bool
            Whether the default layout if channels-first or not

        Returns
        -------
        Tuple[NetworkContext, bool]
            Updated NetworkContext and boolean to indicate success or
            failure


        """

        ioParse = True

        # iterate through all possible mappings and return the first that works
        for idx, mapper in enumerate(self.maps):

            if mapper in self.discardedMappers:
                log.debug(f" ⏭️  Skipping mapper {idx}: {mapper.parser.__class__.__name__} (previously discarded)")
                continue

            newCtxt = ctxt.copy()

            newCtxt, ret = mapper._parse(newCtxt, self.node, default_channels_first, ioParse)

            ioParse = not ret

            if not ret:
                self.discardedMappers.add(mapper)
                continue

            self.mapper = mapper

            # Perform broadcasting
            self.broadcast(newCtxt, default_channels_first)

            newCtxt, ret = mapper._parseCtxt(newCtxt, self.node, default_channels_first)

            if not ret:
                log.debug(f" {FAILURE_MARK} Context parsing failed for {mapper.parser.__class__.__name__}")
                self.discardedMappers.add(mapper)
                continue

            self.mapper.parser.operatorRepresentation['nodeOp'] = self.node.op
            self.mapper.parser.operatorRepresentation['nodeName'] = self.node.name

            return newCtxt, True

        log.debug(f" All {len(self.maps)} mappers exhausted for '{self.node.name}'")
        return ctxt, False

    def _broadcastToNpType(self, ty: Type[BaseType]):

        def _broadcastInteger(ty: Type[IntegerImmediate]):
            if ty.signed:
                return np.dtype(getattr(np, "int" + str(ty.typeWidth)))
            else:
                return np.dtype(getattr(np, "uint" + str(ty.typeWidth)))

        def _broadcastFloat(ty: Type[FloatImmediate]):
            return np.dtype(getattr(np, "double"))

        if issubclass(ty, Pointer) and hasattr(ty, "referencedType"):
            if issubclass(ty.referencedType, IntegerImmediate):
                return _broadcastInteger(ty.referencedType)
        elif issubclass(ty, IntegerImmediate):
            return _broadcastInteger(ty)
        elif issubclass(ty, FloatImmediate):
            return _broadcastFloat(ty)

        return None

    def typeCheck(self, ctxt: NetworkContext) -> Tuple[NetworkContext, bool]:
        """Invokes the mapper's typeCheck method

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext

        Returns
        -------
        Tuple[NetworkContext, bool]
            Updated NetworkContext and boolean to indicate success or
            failure

        """
        if not hasattr(self, 'mapper') or self.mapper is None:
            log.debug(f" {FAILURE_MARK} ONNXLayer.typeCheck() - No mapper selected for '{self.node.name}'")
            return ctxt, False

        newCtxt = ctxt.copy()
        newCtxt, ret = self.mapper.typeCheck(newCtxt, self.node)

        if ret:
            return newCtxt, True

        return ctxt, ret

    def bind(self, ctxt: NetworkContext) -> Tuple[NetworkContext, bool]:
        """Attempt to bind the mapper; discard mapper if binding does not work

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext

        Returns
        -------
        Tuple[NetworkContext, bool]
            Updated NetworkContext and boolean to indicate success or
            failure
        """

        newCtxt = ctxt.copy()
        newCtxt, _, ret = self.mapper.bind(newCtxt, self.node)

        if ret:
            # Update onnx graph with name of the template class
            self.node.attrs['mapping'] = str(self.mapper.binder.template.__class__).split("'")[1]

            # Update shapes and types of tensors in onnx graph based on type inference after binding
            for node in (self.node.inputs + self.node.outputs):
                if ctxt.is_local(node.name):
                    node.shape = ctxt.localObjects[node.name].shape
                    npType = self._broadcastToNpType(ctxt.localObjects[node.name]._type)
                    if npType is not None:
                        node.dtype = npType
                elif ctxt.is_global(node.name):
                    npType = self._broadcastToNpType(ctxt.globalObjects[node.name]._type)
                    if isinstance(ctxt.globalObjects[node.name], ConstantBuffer):
                        if isinstance(node, gs.Constant):
                            node.values = node.values.astype(npType)
                    else:
                        node.shape = ctxt.globalObjects[node.name].shape
                        if npType is not None:
                            node.dtype = npType

            # WIESEP: Compute number of ops only after binding.
            self.mapper.parser.operatorRepresentation['nodeOps'] = int(self.computeOps())
            return newCtxt, True

        self.discardedMappers.add(self.mapper)
        return ctxt, False

    def codeTransform(self, ctxt: NetworkContext, verbose: CodeGenVerbosity = _NoVerbosity) -> NetworkContext:
        """Apply CodeTransformations to associated mapper's binder

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        verbose : CodeGenVerbosity
            CodeGenVerbosity object to control verbosity of generated
            code
        Returns
        -------
        Tuple[NetworkContext, bool]
            Updated NetworkContext

        """
        newCtxt = self.mapper.binder.codeTransform(ctxt, verbose)
        return newCtxt

    def generate(self, ctxt: NetworkContext) -> Tuple[NetworkContext, List[str]]:
        """Invoke mapper's generate method

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext

        Returns
        -------
        Tuple[NetworkContext, List[str]]
            Updated NetworkContext and flag to indicate success

        """

        call = self.mapper.generate(ctxt)

        generated_code = [call]
        return (ctxt, generated_code)


class TopologyOptimizationPass():
    """Abstract pass object which modifies an ONNX graph

    """

    def __init__(self):
        pass

    def apply(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        """Applies a transformation to a graph

        Parameters
        ----------
        graph : gs.Graph
            The neural network being deployed

        Returns
        -------
        Tuple[gs.Graph]
            A modified version of the neural network graph

        """
        return graph


class TopologyOptimizer():
    """Wrapper object to apply multiple TopologyOptimizationPasses sequentially

    """

    def __init__(self, passes: List[TopologyOptimizationPass], name: str = "TopologyOptimizer"):
        self.name = name
        self.passes = passes

    def optimize(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        """Applies passes sequentially

        Parameters
        ----------
        graph : gs.Graph
            Current neural network graph

        Returns
        -------
        Tuple[gs.Graph]
            Modified neural network graph

        """
        for _pass in self.passes:
            start_time = time.perf_counter()
            graph = _pass.apply(graph)
            graph.cleanup().toposort()
            end_time = time.perf_counter()
            log.debug(f" - Applied {_pass.__class__.__name__} ({(end_time - start_time)*1E3:.3f} ms)")
        return graph


class NetworkOptimizationPass(TopologyOptimizationPass):
    """Pass to update the NetworkContext and Neural Network Graph in one go
    """

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        """The method to update context and graph

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        graph : gs.Graph
            Current Neural Network graph

        Returns
        -------
        Tuple[NetworkContext, gs.Graph]:
            Updated context and graph

        """
        return ctxt, graph


class NetworkOptimizer(TopologyOptimizer):
    """Wrapper class to run multiple NetworkOptimizationPasses sequentially
    """

    def optimize(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:  # type: ignore
        """Apply passes sequentially

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        graph : gs.Graph
            Current Neural Network graph

        Returns
        -------
        Tuple[NetworkContext, gs.Graph]: # type: ignor
            Update context and graph

        """
        for _pass in self.passes:
            ctxt, graph = _pass.apply(ctxt, graph)  # type: ignore
            graph.cleanup().toposort()
        return ctxt, graph


class CodeTransformationPass():
    """Pass Object to update code generation; may either modify an executionBlock's existing code snippets or add new code snippets to an executionBlock
    """

    def __init__(self):
        pass

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """Apply the CodeTransformation to an ExecutionBlock

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        executionBlock : ExecutionBlock
            ExecutionBlock whose code you'd like to transform
        name : str
            Graph node name of the operator being targetted
        verbose : CodeGenVerbosity
            Control the verbosity of code generation

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            Updated NetworkContext and ExecutionBlock

        """
        return ctxt, executionBlock


class CodeTransformation():
    """Wrapper object to run multiple CodeTransformations sequentially

    """

    def __init__(self, passes: List[CodeTransformationPass]):
        self.passes = passes

    def transform(self,
                  ctxt: NetworkContext,
                  executionBlock: ExecutionBlock,
                  name: str,
                  verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """Apply passes sequentially to a single ExecutionBlock

        Parameters
        ----------
        ctxt : NetworkContext
            Current NetworkContext
        executionBlock : ExecutionBlock
            ExecutionBlock whose code you'd like to transform
        name : str
            Graph node name of the operator being targetted
        verbose : CodeGenVerbosity
            Control the verbosity of code generation

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            Updated NetworkContext and ExecutionBlock

        """
        for _pass in self.passes:
            ctxt, executionBlock = _pass.apply(ctxt, executionBlock, name, verbose)
        return ctxt, executionBlock


class DeploymentEngine():
    """Deeploy abstraction to represent a compute engine without a complete host system, like an accelerator

    """

    def __init__(self,
                 name: str,
                 Mapping: Dict[str, Union[ONNXLayer, Callable[[gs.Node], Any]]],
                 initCode: str = "",
                 includeList: List[str] = [""]) -> None:
        """Instantiate a new engine

        Parameters
        ----------
        name : str
            Name of this compute engine; must be unique per deployemnt
        Mapping : Dict[str, Union[ONNXLayer, Callable[[gs.Node], Any]]]
            Mapping between operator names and ONNXLayer implementations
        initCode : str
            Static initialization code for this engine
        includeList : List[str]
            List of header files to be included with `#include` directives

        """
        self.name = name  #: str: Name of this compute engine; must be unique per deployemnt
        self.Mapping = Mapping  #: Mapping between operator names and ONNXLayer implementations
        self.initCode = initCode  # str: Static initialization code for this engine
        self.includeList = includeList  #: List[str]: List of header files to be included with `#include` directives

    def canExecute(self, node: gs.Node) -> bool:
        """Return whether this accelerator can execute an operator

        Parameters
        ----------
        node : gs.Node
            Operator to be checked

        Returns
        -------
        bool
            True if operator can be run on this Engine, False
            otherwise

        """
        return node.op in self.Mapping

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', mappings={list(self.Mapping.keys())})"


class DeploymentPlatform():
    """Deeploy abstraction for a complete system, including at least a host core capable of memory allocation

    """

    def __init__(self, engines: List[DeploymentEngine], variableBuffer: Type[VariableBuffer],
                 constantBuffer: Type[ConstantBuffer], structBuffer: Type[StructBuffer],
                 transientBuffer: Type[TransientBuffer]) -> None:
        """Initializes a new deployment platform

        Parameters
        ----------
        engines : List[DeploymentEngine]
            List of all available non-host engines
        variableBuffer : Type[VariableBuffer]
            VariableBuffer subclass with correctly set allocation and
            deallocation templates
        constantBuffer : Type[ConstantBuffer]
            ConstantBuffer subclass with correctly set allocation and
            deallocation templates
        structBuffer : Type[StructBuffer]
            StructBuffer subclass with correctly set allocation and
            deallocation templates
        transientBuffer : Type[TransientBuffer]
            TransientBuffer subclass with correctly set allocation and
            deallocation templates

        """
        assert len(engines) == len(set(engines)), "Duplicate engines are not allowed."
        self.engines = engines  #: List[DeploymentEngine]: A list of all available non-host engines
        self.VariableBuffer = variableBuffer
        self.ConstantBuffer = constantBuffer
        self.StructBuffer = structBuffer
        self.TransientBuffer = transientBuffer

    def __repr__(self) -> str:
        retStr = f"{self.__class__.__name__}("
        retStr += f"engines={[e.name for e in self.engines]}, "
        retStr += f"variableBuffer={self.VariableBuffer.__name__}, "
        retStr += f"constantBuffer={self.ConstantBuffer.__name__}, "
        retStr += f"structBuffer={self.StructBuffer.__name__}, "
        retStr += f"transientBuffer={self.TransientBuffer.__name__}"
        retStr += ")"
        return retStr


class NetworkContainer():
    """Deeploy abstraction for containing the information needed to describe a complete neural network to be deployed

    """

    def __init__(self,
                 graph: gs.Graph,
                 platform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 scheduler: Callable[[gs.Graph], Schedule] = lambda graph: list(graph.nodes),
                 name: str = 'DeeployNetwork',
                 deeployStateDir: str = "DeeployState"):
        """Initializes a new NetworkContainer and its NetworkContext

        Parameters
        ----------
        graph : gs.Graph
            Neural network graph to be deployed
        platform : DeploymentPlatform
            DeploymentPlatform being targetted
        inputTypes : Dict[str, Type[Pointer]]
            DataType for each global network input
        scheduler : Callable[[gs.Graph], Schedule]
            Callable that ingests the graph and returns a list of
            operators to execute
        name : str
            Prefix to use in deployment to uniquify tensor names
        deeployStateDir : str
            Path to a directory to dump intermediate outputs


        """
        self.graph = graph
        self.scheduler = scheduler
        self.layerBinding: 'OrderedDict[str, ONNXLayer]' = OrderedDict()
        self.parsed = False
        self.Platform = platform
        for engine in self.Platform.engines:
            engine.Mapping['Constant'] = lambda x: \
                self.ctxt.hoistConstant(x.attrs['value'], x.outputs[0].name, None)

        self.inputTypes = inputTypes
        self.name = name

        self.ctxt = NetworkContext(variableBuffer = self.Platform.VariableBuffer,
                                   constantBuffer = self.Platform.ConstantBuffer,
                                   structBuffer = self.Platform.StructBuffer,
                                   transientBuffer = self.Platform.TransientBuffer)

        self.deeployStateDir = deeployStateDir

        self.bound = False
        self.transformed = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', platform={self.Platform.__class__.__name__}, inputTypes={ [v.typeName for k, v in self.inputTypes.items()]}, scheduler={self.scheduler.__name__})"

    # Don't override this
    def _createIOBindings(self, ctxt: NetworkContext, graph: gs.Graph):

        for node in graph.inputs:
            data_name = node.name
            data_size = node.shape
            data_type = self.inputTypes[node.name]
            nb = ctxt.VariableBuffer(data_name, data_size)
            nb.is_input = True

            ctxt.add(nb, 'global')
            ctxt.annotateType(data_name, data_type)

        for node in graph.outputs:
            data_name = node.name
            data_size = node.shape
            # WIESEP: The shape and type will be parsed from the graph
            nb = ctxt.VariableBuffer(data_name, data_size)
            nb.is_output = True
            ctxt.add(nb, 'global')

        return ctxt

    def inputs(self) -> List[VariableBuffer]:
        """Return a list of all VariableBuffers that are also global inputs of the network

        Returns
        -------
        List[VariableBuffer]
            Global inputs

        """
        inputs = []

        for tensor in self.graph.inputs:
            if self.ctxt.is_global(tensor.name):
                buffer = self.ctxt.lookup(tensor.name)
                if isinstance(buffer, self.ctxt.VariableBuffer) and len(buffer._users) > 0:
                    inputs.append(buffer)

        return inputs

    def outputs(self) -> List[VariableBuffer]:
        """Return a list of all VariableBuffers that are also global outputs of the network

        Returns
        -------
        List[VariableBuffer]
            Global outputs

        """
        outputs = []

        for tensor in self.graph.outputs:
            if self.ctxt.is_global(tensor.name):
                buffer = self.ctxt.lookup(tensor.name)
                if isinstance(buffer, self.ctxt.VariableBuffer):
                    outputs.append(buffer)

        return outputs

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        """Apply code transformations on every layer's execution block

        Parameters
        ----------
        verbose : CodeGenVerbosity
            Control code generation verbosity

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if the entire network is not bound

        """
        if not self.bound:
            raise RuntimeError('You need to bind the network before transforming code!')

        if self.transformed:
            return

        for name, layer in self.layerBinding.items():
            self.ctxt = layer.codeTransform(self.ctxt, verbose)
        self.transformed = True

    def _selectEngine(self, node: gs.Node) -> DeploymentEngine:
        for engine in self.Platform.engines:
            if node.op in engine.Mapping:
                return engine
        raise RuntimeError(f"No mapping found for node {node.name} with op type {node.op}")

    def _bindLayers(self):
        # Create schedule, binding, then parse resulting program for correctness
        self.layerBinding: 'OrderedDict[str, ONNXLayer]' = OrderedDict()

        schedule = self.scheduler(self.graph)
        flatSchedule = []

        for subGraph in schedule:
            if isinstance(subGraph, gs.Node):
                flatSchedule.append(subGraph)
            else:
                flatSchedule += subGraph

        for node in flatSchedule:
            engine = self._selectEngine(node)
            layer = engine.Mapping[node.op](node)
            if isinstance(layer, ONNXLayer):
                log.debug(f"   {SUCCESS_MARK} Bind {node.name} to layer {layer.__class__.__name__}")
                self.layerBinding[layer.node.name] = layer

    def _parseNode(self, node: ONNXLayer, ctxt: NetworkContext,
                   default_channels_first: bool) -> Tuple[NetworkContext, bool]:
        newCtxt, parsePass = node.parse(ctxt.copy(), default_channels_first)

        if not parsePass:
            return ctxt, False

        return newCtxt, True

    def _typeCheckNode(self, node: ONNXLayer, ctxt: NetworkContext) -> Tuple[NetworkContext, bool]:
        newCtxt, LayerBindSuccess = node.typeCheck(ctxt)

        if not LayerBindSuccess:
            return ctxt, False

        return newCtxt, True

    # Don't override this
    def parse(self, default_channels_first: bool = True) -> bool:
        """Parses the full network by iteratively exploring mapping and binding options with backtracking

        Parameters
        ----------
        default_channels_first : bool
            Whether the default data layout is CxHxW or HxWxC

        Returns
        -------
        bool
            Returns a boolean to indicate whether parsing was
            successful

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if backtracking was exhausted
            without finding a mapping solution

        """

        self.ctxt = NetworkContext(variableBuffer = self.Platform.VariableBuffer,
                                   constantBuffer = self.Platform.ConstantBuffer,
                                   structBuffer = self.Platform.StructBuffer,
                                   transientBuffer = self.Platform.TransientBuffer)

        log.debug(" - Create IO Bindings")
        self.ctxt = self._createIOBindings(self.ctxt, self.graph)

        log.debug(" - Bind Nodes to Layers")
        self._bindLayers()

        ctxt = self.ctxt.copy()

        ctxtStack = deque()
        scheduledLayerList = list(self.layerBinding.values())
        idx: int = 0

        deepestIdx = 0

        log.debug(" - Parse and Type Check Network")
        start_time = time.perf_counter()

        iteration_main = 0
        iteration_sub = 0
        iteration_tot = 0
        while (idx < len(scheduledLayerList)):
            currentLayer = scheduledLayerList[idx]

            # Log current exploration state
            if idx == 0:
                iteration_main += 1
                iteration_tot += 1
                iteration_sub = 0
                log.debug(31 * "-" + f" MAIN ITERATION {iteration_main:<2} " + 31 * "-")

            log.debug(f"[Layer {idx}] Trying '{currentLayer.node.name}' (op: {currentLayer.node.op})")

            stCtxt = copy.deepcopy(ctxt)

            newCtxt, parseSuccess = self._parseNode(currentLayer, ctxt, default_channels_first)

            typeCheckSuccess = False
            if parseSuccess:
                newCtxt, typeCheckSuccess = self._typeCheckNode(currentLayer, newCtxt)

            if parseSuccess and typeCheckSuccess:
                # SCHEREMO: Continue depth-first exploration
                ctxtStack.append(stCtxt)
                ctxt = newCtxt
                idx = idx + 1
                if idx > deepestIdx:
                    deepestIdx = max(idx, deepestIdx)
                    deepestCtxt = stCtxt

            else:
                # SCHEREMO: If we can't find a mapping for the root, we must exit
                if idx == 0:
                    deepestLayer = scheduledLayerList[deepestIdx]
                    deepestNodeName = deepestLayer.node.name
                    log.debug("-" * 80)
                    log.error("💥 PARSING FAILED - Backtracking exhausted at root!")
                    log.error("=" * 80)
                    log.error(f"🔍 Diagnosis:")
                    log.error(f"   - Deepest successful exploration: Layer {deepestIdx} '{deepestNodeName}'")
                    log.error(
                        f"   - Deepest layer available mappers: {[type(x.parser).__name__ for x in deepestLayer.maps]}")
                    log.error("=" * 80)
                    raise RuntimeError(
                        f'Did not find adequate mapping for graph! Explored until layer {deepestLayer.__class__.__name__} of node {deepestNodeName}'
                        f'Candidates: {[type(x.parser).__name__ for x in deepestLayer.maps]}. Exhausted backtracking.')

                previousLayer = scheduledLayerList[idx - 1]
                ctxt = ctxtStack.pop()

                # Keep options of current layer open - the upstream mapping will change, so we don't know which options are feasible here
                currentLayer.resetDiscardedMappers()

                # Update the previous layer, by discarding the current mapper or binder
                if previousLayer.mapper.bindingsExhausted():
                    previousLayer.discardCurrentMapper()
                else:
                    previousLayer.mapper.discardCurrentBinder()

                # SCHEREMO: Rollback one step
                idx = idx - 1
                if idx != 0:
                    iteration_sub += 1
                    iteration_tot += 1
                    log.debug(31 * "-" + f" SUB ITERATION {iteration_main}.{iteration_sub:<2} " + 31 * "-")

        end_time = time.perf_counter()
        log.info(
            f" {SUCCESS_MARK} Parsed network with {len(self.layerBinding)} layers after {iteration_tot} iterations in {(end_time-start_time)*1E3:.3f} ms"
        )
        self.ctxt = ctxt
        self.parsed = True
        return True

    def bind(self) -> bool:
        """Bind the entire network layer-by-layer

        Returns
        -------
        bool
            Return true if binding was successful

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if the network has not been parsed
            of there exists no valid binding

        """
        if not self.parsed:
            raise RuntimeError('You need to parse the network before binding!')

        # SCHEREMO: Implement backtracking here! Currently tries the cheapest branch only!
        newCtxt = self.ctxt.copy()

        NetworkBindSuccess = True
        log.info("- Map Layers to Bindings")
        for name, layer in self.layerBinding.items():

            newCtxt, LayerBindSuccess = layer.bind(newCtxt)
            NetworkBindSuccess = NetworkBindSuccess and LayerBindSuccess

            if not NetworkBindSuccess:
                raise RuntimeError(f'Could not find a valid binding for the graph')

            log.debug(f" {SUCCESS_MARK} Mapped {layer.node.name} to {layer.mapper.binder}")

        self.bound = True
        self.ctxt = newCtxt

        return True

    # Don't override this
    def generateInferenceCode(self) -> str:
        """Generate the actual inference function for the entire network

        Returns
        -------
        str
            The full inference method

        Raises
        ------
        ValueError
            Raises a RuntimeError if network is not parsed and bound

        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before generating code!')

        callStack = ''

        for key, node in self.layerBinding.items():
            self.ctxt, code = node.generate(self.ctxt)

            sections = reduce(lambda a, b: a + b, code, [])
            callStack += reduce(lambda a, b: a + b, sections, "")

        return callStack

    # Don't override this
    def generateGlobalDefinitionCode(self) -> str:
        """Generate all global definition code for inference

        Returns
        -------
        str
            Global Definition code

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if network is not parsed and bound

        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before generating code!')

        callStack = reduce(
            lambda a, b: a + b,
            [obj.definition for obj in self.ctxt.globalObjects.values() if isinstance(obj, GlobalDefinition)], "")

        return callStack

    # Don't override this
    def generateInferenceInitializationCode(self) -> str:
        """Generate initialization code, including static memory allocation and other setup tasks

        Returns
        -------
        str
            Initialization code

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if network is not parsed and bound

        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before generating code!')

        callStack = ''
        for node in self.ctxt.localObjects.values():
            # WIESEP: We don't want to initialize the struct buffers as this should be handled by the ArgumentStructGeneration
            if isinstance(node, StructBuffer):
                continue

            name = node.name
            node.name = self.ctxt._mangle(node.name)
            callStack += node.init()
            node.name = name

        return callStack

    # Don't override this
    def generateIOBufferInitializationCode(self) -> str:
        """Generate initialization code for global network inputs and outputs

        Returns
        -------
        str
            Initialization code

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if network is not parsed and bound

        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before generating code!')

        callStack = ''
        inputNum = 0
        outputNum = 0
        inputs = self.inputs()
        outputs = self.outputs()

        for node in self.ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, (StructBuffer, ConstantBuffer)):
                assert issubclass(node._type, Pointer), f"IO Buffer {node.name} is not a Pointer!"
                if node._deploy:
                    name = node.name
                    node.name = self.ctxt._mangle(node.name)
                    callStack += "extern " + node.init()
                    # SCHEREMO: Borderline hacky, but on the okay side of things, I think
                    callStack += "static const uint32_t " + node.name + "_len" + " = " + str(np.prod(node.shape)) + ";"
                    node.name = name

        callStack += "static const uint32_t " + self.ctxt._mangle("num_inputs") + f" = {len(inputs)};"
        callStack += "static const uint32_t " + self.ctxt._mangle("num_outputs") + f" = {len(outputs)};"

        callStack += "extern void* " + self.ctxt._mangle("inputs") + f"[{len(inputs)}];"
        callStack += "extern void* " + self.ctxt._mangle("outputs") + f"[{len(outputs)}];"

        callStack += "static const uint32_t " + self.ctxt._mangle("inputs_bytes") + f"[{len(inputs)}] = " + "{"

        numBytes = []
        for node in inputs:
            numBytes.append(str(np.prod(node.shape) * node._type.referencedType.typeWidth // 8))
        callStack += ", ".join(numBytes)

        callStack += "};"

        callStack += "static const uint32_t " + self.ctxt._mangle("outputs_bytes") + f"[{len(outputs)}] = " + "{"

        numBytes = []
        for node in outputs:
            numBytes.append(str(np.prod(node.shape) * node._type.referencedType.typeWidth // 8))
        callStack += ", ".join(numBytes)

        callStack += "};"

        return callStack

    @property
    def worstCaseBufferSize(self):
        """Return the worst-case buffer size occupied by the network implementaiton
        """
        return self.ctxt._maxDynamicSize

    # Don't override this
    def generateBufferInitializationCode(self) -> str:
        """Generates code for all forward-declaration of buffers used during inference

        Returns
        -------
        str
            Returns forward-declaration code

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if network is not parsed and bound

        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()

        inputs = self.inputs()
        outputs = self.outputs()

        callStack = ''
        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                assert issubclass(node._type, Pointer), f"Global VariableBuffer {node.name} is not a Pointer!"
                if node._deploy:
                    name = node.name
                    node.name = ctxt._mangle(node.name)
                    callStack += node.init()
                    node.name = name

        for node in ctxt.globalObjects.values():
            if isinstance(node, StructBuffer):
                name = node.name
                node.name = ctxt._mangle(node.name)
                callStack += node.init()
                node.name = name

        callStack += "void* " + ctxt._mangle("inputs") + f"[{len(inputs)}];"
        callStack += "void* " + ctxt._mangle("outputs") + f"[{len(outputs)}];"

        return callStack

    def generateBufferAllocationCode(self) -> str:
        """Generates code to allocate space for the global input and output buffer of the network

        Returns
        -------
        str
            Allocation code for global IO buffers

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if network is not parsed and bound


        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()

        inputs = self.inputs()
        outputs = self.outputs()
        callStack = ''

        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                assert issubclass(node._type, Pointer), f"Global VariableBuffer {node.name} is not a Pointer!"
                if node._deploy:
                    name = node.name
                    node.name = ctxt._mangle(node.name)
                    callStack += node.alloc()
                    node.name = name

        for node in ctxt.globalObjects.values():
            if isinstance(node, StructBuffer):

                if node._deploy:
                    name = node.name
                    node.name = ctxt._mangle(node.name)
                    callStack += node.alloc()
                    node.name = name

        for idx, i in enumerate(inputs):
            callStack += ctxt._mangle("inputs") + f"[{idx}] = (void*) {ctxt._mangle(i.name)};"
        for idx, i in enumerate(outputs):
            callStack += ctxt._mangle("outputs") + f"[{idx}] = (void*) {ctxt._mangle(i.name)};"

        return callStack

    # Don't override this
    def generateBufferDeAllocationCode(self) -> str:
        """Generates code to deallocate all global buffers

        Returns
        -------
        str
            Code to deallocate buffers

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if network is not parsed and bound



        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before generating code!')

        callStack = ''
        for node in self.ctxt.globalObjects.values():
            if node._deploy:
                node.name = self.ctxt._mangle(node.name)
                callStack += node.dealloc()

        return callStack

    def generateIncludeString(self) -> str:
        """Generate code to include platform-dependent includes

        Returns
        -------
        str
            Include code

        """
        includeStr = []
        for engine in self.Platform.engines:
            for include in engine.includeList:
                includeStr += ["#include \"" + include + "\""]
        return ("\n").join(includeStr)

    def generateEngineInitializationCode(self) -> str:
        """Generate initialization code for all compute engines

        Returns
        -------
        str
            Initialization code for all engines

        """
        return ("\n").join([engine.initCode for engine in self.Platform.engines])

    def numberOfOps(self, verbose: bool) -> int:
        """Returns the total number of operations per network inference

        Parameters
        ----------
        verbose : bool
            Control whether the number of operations are printed to
            STDOUT for each operator

        Returns
        -------
        int
            Number of operations (1 MAC = 2 Ops) per network inference

        Raises
        ------
        RuntimeError
            Raises a RuntimeError if network is not parsed and bound


        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before getting number of operations!')
        totalSum = 0
        for i in self.layerBinding.values():
            nodeOps = i.mapper.parser.operatorRepresentation['nodeOps']
            totalSum += nodeOps
            if verbose:
                log.info(f"Layer '{i.node.name}'")
                log.info(f"  Number of Operations        : {nodeOps}")
        return totalSum

        # Don't override this
    def _exportGraph(self, folderPath, fileName):
        relativeDataPath = os.path.join(folderPath, fileName + _dataExtension)
        absoluteDataPath = os.path.abspath(relativeDataPath)
        relativeOnnxPath = os.path.join(folderPath, fileName + _graphExtension)
        absoluteOnnxPath = os.path.abspath(relativeOnnxPath)

        if not os.path.isabs(absoluteOnnxPath) or not os.path.isabs(absoluteDataPath):
            raise OSError(f"Error exporting the context to: {absoluteOnnxPath}")

        # VJUNG: ONNX-Graphsurgeon needs tensors to be in their export types
        constTensors = [tensor for tensor in self.graph.tensors().values() if isinstance(tensor, gs.Constant)]
        for tensor in constTensors:
            if tensor.dtype != tensor.export_dtype:
                tensor.values = tensor.values.astype(tensor.export_dtype)

        model = gs.export_onnx(self.graph)

        # Annotate additional information in doc_string of tensors
        for tensor in (list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input) +
                       list(model.graph.initializer)):
            if tensor.name in self.ctxt.localObjects:
                lObject = self.ctxt.localObjects[tensor.name]
                tensor.doc_string += f"Biased: {lObject._signed}, "
                tensor.doc_string += f"nLevels: {lObject.nLevels}, "
                tensor.doc_string += f"Deeploy: {lObject._deploy}, "
                if not isinstance(lObject, ConstantBuffer) and hasattr(lObject, "_type"):
                    tensor.doc_string += f"Type: {lObject._type.typeName}, "
                    if hasattr(lObject._type, "referencedType"):
                        tensor.doc_string += f"Reference Type: {lObject._type.referencedType.typeName}"
            elif tensor.name in self.ctxt.globalObjects:
                gObject = self.ctxt.globalObjects[tensor.name]
                tensor.doc_string += f"Biased: {gObject._signed}, "
                tensor.doc_string += f"nLevels: {gObject.nLevels}, "
                tensor.doc_string += f"Deeploy: {gObject._deploy}, "
                if not isinstance(gObject, ConstantBuffer) and hasattr(gObject, "_type"):
                    tensor.doc_string += f"Type: {gObject._type.typeName}, "
                    if hasattr(gObject._type, "referencedType"):
                        tensor.doc_string += f"Reference Type: {gObject._type.referencedType.typeName}"

        convert_model_to_external_data(model, location = fileName + _dataExtension)
        onnx.save(model, absoluteOnnxPath)

    def exportDeeployState(self, folderPath: str, fileName: str):
        """Export compressed network context and neural network graph

        Parameters
        ----------
        folderPath : str
            path to directory where to save context and graph
        fileName : str
            prefix to use when saving artifacts

        """

        os.makedirs(os.path.abspath(folderPath), exist_ok = True)
        self._exportGraph(folderPath, fileName)
        self.ctxt.exportNetworkContext(folderPath, fileName)

    @staticmethod
    def _importONNXGraph(folderPath: str, fileName: str) -> gs.Graph:
        relativePath = os.path.join(folderPath, fileName + _graphExtension)
        absolutePath = os.path.abspath(relativePath)

        if not os.path.isabs(absolutePath) or not os.path.exists(absolutePath):
            raise OSError(f"File or path does not exist: {absolutePath}")

        onnx_graph = onnx.load_model(absolutePath)
        return gs.import_onnx(onnx_graph)

    def importDeeployState(self, folderPath: str, fileName: str):
        """Override this container's graph and context with loaded compressed artifacts

        Parameters
        ----------
        folderPath : str
            Path to the artifact directory
        fileName : str
            prefix of the saved artifacts

        """
        self.graph = NetworkDeployer._importONNXGraph(folderPath, f"{fileName}")
        self.ctxt = NetworkContext.importNetworkContext(folderPath, f"{fileName}")


class NetworkDeployer(NetworkContainer):
    """Deeploy abstraction to contain an entire network and all necessary information to deploy it
    """

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable[[gs.Graph], Schedule] = lambda graph: list(graph.nodes),
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState"):
        """Initialize a new NetworkDeployer

        Parameters
        ----------
        graph : gs.Graph
            The raw neural network graph to be deployed, e.g. an output
            from Quantlib
        deploymentPlatform : DeploymentPlatform
            The target deployment platform
        inputTypes : Dict[str, Type[Pointer]]
            A mapping of global network inputs to Deeploy datatypes
        loweringOptimizer : TopologyOptimizer
            A topology optimizer used to transform the network into a
            representation that can be mapped to NodeMappers
        scheduler : Callable[[gs.Graph], Schedule]
            Method to topologically sort the graph into the order of
            execution
        name : str
            Prefix to avoid name conflicts between Deeploy code and other
            code
        default_channels_first : bool
            Whether data layout is CxHxW, i.e. channels are first, or
            HxWxC, i.e. channels are last
        deeployStateDir : str
            Directory where intermediate states are saved


        """
        super().__init__(graph, deploymentPlatform, inputTypes, scheduler, name, deeployStateDir = deeployStateDir)

        self.loweringOptimizer = loweringOptimizer
        self.default_channels_first = default_channels_first

        self.prepared = False

    def __repr__(self):
        return super().__repr__(
        ) + f" (loweringOptimizer: {self.loweringOptimizer.name}, default_channels_first: {self.default_channels_first})"

    # Don't override this
    def lower(self, graph: gs.Graph) -> gs.Graph:
        """Apply the lowering optimize

        Parameters
        ----------
        graph : gs.Graph
            Unmodified input neural network graph

        Returns
        -------
        gs.Graph
            Neural network graph that is deployable with the
            DeploymentPlatform's Mapping

        """
        return self.loweringOptimizer.optimize(graph)

    # Helper function for duplicate constants
    # Makes sure to copy the value because otherwise, constant folding fails with a warning
    def _duplicateConstantNode(self,
                               node: gs.Node,
                               name: str,
                               inputs: Optional[List[gs.Tensor]] = None,
                               outputs: Optional[List[gs.Tensor]] = None) -> gs.Node:
        assert node.op == "Constant", f"Expected a Constant node, received node of op {node.op}"
        assert "value" in node.attrs, f"Constant node doesn't have a \"value\" attribute"
        value = node.attrs["value"]
        if isinstance(value, gs.Constant):
            newValue = gs.Constant(f'{value.name}_OF_{name}', value.values.copy())
        elif isinstance(value, np.ndarray):
            newValue = value.copy()
        else:
            assert False, f"Node {node.name}: Unrecognized value type {type(value)}"
        return gs.Node(
            op = "Constant",
            name = name,
            attrs = {"value": newValue},
            inputs = inputs,
            outputs = outputs,
        )

    # Don't override this
    # Duplicate constants with multiple users
    def _duplicateConstants(self, graph: gs.Graph) -> None:
        # Duplicate constant tensors
        for tensor in filter(lambda t: isinstance(t, gs.Constant) and len(t.outputs) > 1, graph.tensors().values()):
            for node in tensor.outputs:
                newConst = tensor.copy()
                newConst.name += f"_DUPLICATE_FOR_{node.name}"
                assert isinstance(node, gs.Node), f"Expected node to be an instance of gs.Node. Received {type(node)}"
                node.inputs.insert(node.inputs.index(tensor), newConst)
            tensor.outputs.clear()

        # Duplicate constant nodes with multiple outputs
        nodes = list(graph.nodes)
        for node in filter(lambda n: n.op == "Constant" and len(n.outputs) > 1, nodes):
            output_tensors = list(node.outputs)
            for tensor in output_tensors:
                assert len(
                    tensor.inputs
                ) == 1, f"Expected output tensor to have only a single input. The tensor has {len(tensor.inputs)}"
                tensor.inputs.clear()
                newConst = self._duplicateConstantNode(
                    node,
                    f"{node.name}_DUPLICATE_FOR_{tensor.name}",
                    outputs = [tensor],
                )
                graph.nodes.append(newConst)
            node.outputs.clear()

        # Duplicate constant nodes which have an output tensor that connects to multiple outputs
        for node in filter(lambda n: n.op == "Constant" and len(n.outputs) == 1 and len(n.outputs[0].outputs) > 1,
                           nodes):
            tensor = node.outputs[0]
            for downstreamNode in tensor.outputs:
                assert isinstance(
                    downstreamNode, gs.Node
                ), f"Expected the downstream node to be an instance of gs.Node. Received {type(downstreamNode)}"
                newTensor = tensor.copy()
                newTensor.name += f"_DUPLICATE_FOR_{downstreamNode.name}"
                newNode = self._duplicateConstantNode(
                    node,
                    f"{node.name}_DUPLICATE_FOR_{downstreamNode.name}",
                    outputs = [newTensor],
                )
                graph.nodes.append(newNode)
                downstreamNode.inputs.insert(downstreamNode.inputs.index(tensor), newTensor)
            tensor.outputs.clear()

        graph.cleanup().toposort()

    def _foldConstants(self, graph: gs.Graph):
        graph.fold_constants()
        graph.cleanup().toposort()

    def _sanitizeGraphNames(self, graph: gs.Graph):

        def sanitize(name: str) -> str:
            # Remove illegal characters: anything not a letter, digit, or underscore.
            sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
            return sanitized_name

        for node in graph.nodes:
            node.name = sanitize(node.name)
            for tensor in node.inputs + node.outputs:
                tensor.name = sanitize(tensor.name)

    # Don't override this
    def _removeEmptyInputs(self, graph: gs.Graph):
        _inps = self.graph.inputs.copy()
        for inp in _inps:
            if np.prod(inp.shape) == 0:
                self.graph.inputs.remove(inp)

    # Don't override this
    def _mangleTensorNames(self):
        """Mangle tensor names

        This adds _tensor suffix to all tensors in hopes to make them distinct from other graph elements, e.g., nodes.
        Deeploy needs tensor names to be distinct for code snippet introspection.
        """
        for tensor in self.graph.tensors().values():
            tensor.name = f"{tensor.name}_tensor"

    # Don't override this
    def _mangleNodeNames(self):
        """Mangle node names only if duplicates exist. Unique names are preserved."""
        # Count occurrences of each original name
        counts: Dict[str, int] = {}
        for node in self.graph.nodes:
            counts[node.name] = counts.get(node.name, 0) + 1

        # For any name that appears more than once, append a counter suffix
        seen: Dict[str, int] = {}
        for node in self.graph.nodes:
            orig = node.name
            if counts[orig] > 1:
                idx = seen.get(orig, 0)
                node.name = f"{orig}_{idx}"
                seen[orig] = idx + 1
            # else: unique name, leave it unchanged

    # Don't override this
    def _removeIdentityNodes(self):
        for node in filter(lambda x: x.op == "Identity", self.graph.nodes):
            self.graph.deleteNode(node)

    def _assertTensorsHaveShape(self) -> None:
        missingShapes = [name for name, tensor in self.graph.tensors().items() if tensor.shape is None]
        assert len(missingShapes) == 0, \
            f"Shape inference is not supported.\nFound tensors with missing shape annotation: {missingShapes}"

    def frontEnd(self):
        """API hook to prepare the graph to be deployed and build the initial NetworkContext

        """

        log.info(80 * "=")
        log.info("Deeploy FrontEnd")
        log.info(80 * "=")

        log.info("- Apply Preprocessing")

        log.debug(" - Remove Identity Nodes")
        self._removeIdentityNodes()

        log.debug(" - Mangle Tensor Names")
        self._mangleTensorNames()

        log.debug(" - Mangle Node Names")
        self._mangleNodeNames()

        # Rename graph inputs and outputs:
        for idx, inputNode in enumerate(self.graph.inputs):
            inputNode.name = "input_" + str(idx)
        for idx, outputNode in enumerate(self.graph.outputs):
            outputNode.name = "output_" + str(idx)

        log.debug(" - Sanitize Graph Names")
        self._sanitizeGraphNames(self.graph)

        log.debug(" - Remove Empty Inputs")
        self._removeEmptyInputs(self.graph)

        log.debug(" - Duplicate Constants")
        self._duplicateConstants(self.graph)

        log.debug(" - Constant Folding")
        self._foldConstants(self.graph)

        log.info(f"> Export State to {_middlewarePreLoweringFilename}[.onnx|.pkl]")
        self.exportDeeployState(self.deeployStateDir, _middlewarePreLoweringFilename)

        log.info("- Perform Graph Lowering")
        self.graph = self.lower(self.graph)  # This lowers the graph to a deployable format

        log.info(f"> Export State {_middlewarePostLoweringFilename}[.onnx|.pkl]")
        self.exportDeeployState(self.deeployStateDir, _middlewarePostLoweringFilename)

        log.info(" - Assert all tensors have a shape annotation")
        self._assertTensorsHaveShape()

        log.info("- Perform Graph Parsing")
        try:
            self.parse(self.default_channels_first)  # This reparses the lowered graph
        except Exception as e:
            log.error(f"Error during parsing! Exporting deeploy state {_backendPostBindingFilename}[.onnx|.pkl]!")
            self.exportDeeployState(self.deeployStateDir, _backendPostBindingFilename)
            raise e

    # Don't Override this
    def midEnd(self):
        """API hook to be used after finalizing kernel selection; hoist transient buffers, and perform low-level code optimizations (e.g. tiling and static memory allocation)
        """
        log.info(80 * "=")
        log.info("Deeploy MidEnd")
        log.info(80 * "=")
        try:
            self.bind()
        except Exception as e:
            log.error("Error during binding! Exporting deeploy state!")
            self.exportDeeployState(self.deeployStateDir, _backendPostBindingFilename)
            raise e

        log.info(f"> Export State {_backendPostParsingFilename}[.onnx|.pkl]")
        self.exportDeeployState(self.deeployStateDir, _backendPostParsingFilename)

    # Don't override this unless you know what you are doin
    def backEnd(self, verbose: CodeGenVerbosity = _NoVerbosity):
        """API hook to generate code once kernel implementations are picked and tiling, memory allocation, and other low-level optimizations have been done.

        Parameters
        ----------
        verbose : CodeGenVerbosity
            Control verbosity of generated code

        """
        log.info(80 * "=")
        log.info("Deeploy BackEnd")
        log.info(80 * "=")

        log.info("- Performing code transformations and optimization...")
        self.codeTransform(verbose)

        log.info(f"> Export State {_backendPostBindingFilename}[.onnx|.pkl]")
        self.exportDeeployState(self.deeployStateDir, _backendPostBindingFilename)

    # Don't override this
    def prepare(self, verbose: CodeGenVerbosity = _NoVerbosity):
        """API hook to perform the entire deployment process to the point where generated code may be extracted

        Parameters
        ----------
        verbose : CodeGenVerbosity
            Control verbosity of generated code

        """
        self.frontEnd()

        self.midEnd()

        self.backEnd(verbose = verbose)
        self.prepared = True

    def _printInputOutputSummary(self):
        log.info("Input:")
        for buf in self.inputs():
            log.info(f" - '{buf.name}': Type: {buf._type.referencedType.typeName}")

        log.info('Output:')
        for buf in self.outputs():
            log.info(f" - '{buf.name}': Type: {buf._type.referencedType.typeName}")

    def _printMemorySummary(self):
        log.info("")
        log.info("Memory Usage Report:")
        log.info(f"  Level                 Total (bytes)   (Static + Dynamic)    ")
        log.info("  " + "-" * 60)

        _worstCaseBufferSize = self.worstCaseBufferSize
        if len(_worstCaseBufferSize) == 0:
            _worstCaseBufferSize = {"None": 0}

        for level, dynamicSize in _worstCaseBufferSize.items():
            staticSize = 0
            for _buffer in self.ctxt.globalObjects.values():
                # We do not count structs for now, since they are not properly modeled
                if isinstance(_buffer, ConstantBuffer) or (isinstance(_buffer, VariableBuffer) and _buffer._deploy):
                    # SCHEREMO: We only
                    if (hasattr(_buffer, "_memoryLevel") and _buffer._memoryLevel == level) or level == "None":
                        staticSize += int((np.prod(_buffer.shape) * _buffer._type.referencedType.typeWidth // 8))
                    else:
                        log.warning(f"Buffer {_buffer.name} does not have a valid memory level")

            total = staticSize + dynamicSize

            log.info(f"  {level:<22}     {total:8,d}   "
                     f"({staticSize:6,d} + {dynamicSize:7,d})  ")

    def generateFunction(self, verbose: CodeGenVerbosity = _NoVerbosity) -> str:
        """Helper function to prepare deployment and return generated function code

        """

        if not self.prepared:
            self.prepare(verbose = verbose)

        log.info("=" * 80)
        log.info("Deeploy Code Generation")
        log.info("=" * 80)

        self._printInputOutputSummary()

        num_ops = self.numberOfOps(verbose = True)
        log.info("-" * 80)

        log.info(f"Number of Ops.                : {num_ops}")

        self._printMemorySummary()

        return self.generateInferenceCode()
