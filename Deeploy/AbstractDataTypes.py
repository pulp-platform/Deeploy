# ----------------------------------------------------------------------
#
# File: AbstractDataTypes.py
#
# Last edited: 25.04.2023
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

from __future__ import annotations

import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union

import numpy as np

_NetworkContext = TypeVar("_NetworkContext")

_PointerType = TypeVar("Pointer", bound = "Pointer")
_ImmediateType = TypeVar("Immediate", bound = "Immediate")
_StructType = TypeVar("Struct", bound = "Struct")

_DeeployType = TypeVar("_DeeployType", _PointerType, _ImmediateType, _StructType)
_PythonType = TypeVar("_PythonType", str, int, float, Dict[str, "_PythonType"], Iterable["_PythonType"])


class _ClassPropertyDescriptor(object):

    def __init__(self, fget, fset = None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, other = None):
        if other is None:
            other = type(obj)
        return self.fget.__get__(obj, other)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def _classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return _ClassPropertyDescriptor(func)


class _SlotPickleMixin(object):

    def __getstate__(self):
        return dict((slot, getattr(self, slot)) for slot in self.__slots__ if hasattr(self, slot))

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)


@dataclass
class BaseType(Generic[_PythonType, _DeeployType], _SlotPickleMixin):
    """Deeploy abstraction to represent data types that can be expressed in the C language
    """

    __slots__ = [
        "value"  #: _PythonType: Variable that stores the underlying represented Python-typed value
    ]
    typeName: str  #: str: The C typename of this type
    typeWidth: int  #: int: the number of BITS to be assigned to the type

    @classmethod
    @abstractmethod
    def checkValue(cls, value: _PythonType, ctxt: Optional[_NetworkContext] = None) -> bool:
        """Checks whether a given Python-type value (usually FP64) can be represented with a Deeploy type

        Parameters
        ----------
        value : _PythonType
            Python-typed value to check
        ctxt : Optional[_NetworkContext]
            Current NetworkContext

        Returns
        -------
        bool
            Returns true if value can represented by cls

        """
        return False

    @classmethod
    @abstractmethod
    def checkPromotion(cls, value: Union[_PythonType, _DeeployType], ctxt: Optional[_NetworkContext] = None) -> bool:
        """Checks whether a given Python-typed or Deeploy-typed value can be represented with the Deeploy type

        Parameters
        ----------
        value : Union[_PythonType, _DeeployType]
            Python-typed or Deeploy-typed value to be checked for
            promotion to cls
        ctxt : Optional[_NetworkContext]
            Current NetworkContext

        Returns
        -------
        bool
            Returns true if the value can be promoted to cls

        """
        return False


class VoidType(BaseType):
    """Helper type to represent the C void type for pointers

    """
    __slots__ = []
    typeName = "void"
    typeWidth = 32


class Immediate(BaseType[_PythonType, _ImmediateType]):
    """Represents any immediate value, e.g. 6, 7.48,... Can not be used to represent values that are deferenced at runtime.
    """

    def __init__(self, value: Union[int, float, Immediate], ctxt: Optional[_NetworkContext] = None):
        assert self.checkPromotion(value), f"Cannot assign {value} to a {self.typeName}"
        self.value = value

    @classmethod
    def partialOrderUpcast(cls, otherCls: Type[Immediate]) -> bool:
        """This method checks whether a data type (cls) can be used to represent any value that can be represented by another data type (otherCls). For more information on partial order sets and type conversion, check:https://en.wikipedia.org/wiki/Partially_ordered_set https://en.wikipedia.org/wiki/Type_conversion

        Parameters
        ----------
        otherCls : Type[Immediate]
            The class you want to upcast an immediate of this cls to

        Returns
        -------
        bool
            Returns true if this cls can be statically promoted to
            otherCls

        """
        return False

    @classmethod
    def checkPromotion(cls, value: Union[_PythonType, _ImmediateType], ctxt: Optional[_NetworkContext] = None):
        # SCHEREMO: np.ndarray is Iterable
        if isinstance(value, Immediate):
            return cls.checkPromotion(value.value, ctxt)

        return cls.checkValue(value, ctxt)

    def __eq__(self, other) -> bool:
        if not (isinstance(self, type(other)) and hasattr(other, "value")):
            return False
        return self.value == other.value

    def __repr__(self) -> str:
        return f"{str(self.value)}"


class IntegerImmediate(Immediate[Union[int, Iterable[int]], _ImmediateType]):

    signed: bool  #: bool: Represents whether the underlying integer is signed or unsigned
    typeMax: int  #: int: Represents the largest possible representable value, i.e. `2^{typeWidth}-1` for unsigned values and `2^{typeWidth-1}-1` for signed values.
    typeMin: int  #: int: Represenst the smallest possible representable value, i.e. `0` for unsigned values and `-2^{typeWidth-1}` for signed values.

    @_classproperty
    def typeMax(cls) -> int:
        if cls.signed:
            return 2**(cls.typeWidth - 1) - 1
        else:
            return 2**(cls.typeWidth) - 1

    @_classproperty
    def typeMin(cls) -> int:
        if cls.signed:
            return -2**(cls.typeWidth - 1)
        else:
            return 0

    @classmethod
    def partialOrderUpcast(cls, otherCls: Type[Immediate]) -> bool:
        if issubclass(otherCls, IntegerImmediate):
            return cls.typeMax >= otherCls.typeMax and cls.typeMin <= otherCls.typeMin
        else:
            return False

    @classmethod
    def checkValue(cls, value: Union[int, Iterable[int]], ctxt: Optional[_NetworkContext] = None):

        if isinstance(value, int):
            _max, _min = (value, value)
        elif isinstance(value, np.ndarray):
            _max = value.max()
            _min = value.min()
        elif isinstance(value, Iterable):
            _max = max(value)
            _min = min(value)

        if _max > cls.typeMax:
            return False
        if _min < cls.typeMin:
            return False
        return True


class Pointer(BaseType[Optional[str], _PointerType]):
    """Represents a C Pointer type to an underlying BaseType data type
    """

    __slots__: List[str] = ["referenceName", "_mangledReferenceName"]
    referencedType: Type[
        _DeeployType]  #: Type[_DeeployType]: type definition of the underlying type that this type points to

    @_classproperty
    def typeName(cls):
        return cls.referencedType.typeName + "*"

    @classmethod
    def checkValue(cls, value: Optional[str], ctxt: Optional[_NetworkContext] = None) -> bool:
        if ctxt is None:
            return False

        if value is None or value == "NULL":
            print("WARNING: Setting pointer value to NULL - Referenced data is invalid!")
            return True

        reference = ctxt.lookup(value)

        if hasattr(reference, "_type") and reference._type is not None:
            # Void pointer & DeeployType check
            _type = reference._type
            if not issubclass(cls.referencedType, VoidType) and _type.referencedType != cls.referencedType:
                return False
            return True

        if not hasattr(reference, value):
            return True
        return cls.referencedType.checkPromotion(reference.value, ctxt)

    @classmethod
    def checkPromotion(cls, _value: Union[Optional[str], Pointer], ctxt: Optional[_NetworkContext] = None) -> bool:
        if isinstance(_value, Pointer):
            value = _value.referenceName
        else:
            value = _value
        return cls.checkValue(value, ctxt)

    def __init__(self, _value: Union[Optional[str], Pointer], ctxt: Optional[_NetworkContext] = None):
        """Initializes a pointer to a registered object in the NetworkContext

        Parameters
        ----------
        _value : Union[Optional[str], Pointer]
            Name of the memory buffer in the NetworkContext to be
            represented or Pointer object
        ctxt : Optional[_NetworkContext]
            Current NetworkContext

        Raises
        ------
        ValueError
            Raises a ValueError if the memory buffer does not exist or
            cannot be pointed to with this Pointer class

        """

        if _value is not None and not self.checkPromotion(_value, ctxt):
            raise ValueError(f"value {_value} is not of type {self.referencedType}!")

        if _value is None:
            self.referenceName = "NULL"  #: str: Either NULL iff this pointer corresponds to a NULL pointer in C, or the name of the memory buffer this pointer points to.
            self._mangledReferenceName = "NULL"
        elif isinstance(_value, Pointer):
            self.referenceName = _value.referenceName
            self._mangledReferenceName = _value._mangledReferenceName
        else:
            self.referenceName = _value
            self._mangledReferenceName = ctxt._mangle(_value)

    def __eq__(self, other):
        if not isinstance(other, Pointer):
            return False

        return self.referenceName == other.referenceName

    def __repr__(self):
        return f"{self._mangledReferenceName}"


class Struct(BaseType[Union[str, Dict[str, _DeeployType]], _StructType]):
    """Deeploy data type abstraction for C-like packed structs
    """

    structTypeDict: Dict[str, Type[BaseType]] = {
    }  #: Dict[str, Type[BaseType]]: The definition of the struct mapping its field names to their associated Deeploy-types

    @_classproperty
    def typeWidth(cls) -> int:
        return sum(q.typeWidth for q in cls.structTypeDict.values())

    @classmethod
    def _castDict(cls,
                  inputValue: Union[str, Struct, Dict[str, BaseType]],
                  ctxt: Optional[_NetworkContext] = None) -> Dict[str, BaseType]:

        if isinstance(inputValue, str):
            inputDict = ctxt.lookup(inputValue).structDict.value
        elif isinstance(inputValue, Struct):
            inputDict = inputValue.value
        else:
            inputDict = inputValue

        castedDict: Dict[str, BaseType] = {}

        for key, value in copy.deepcopy(inputDict).items():
            castedDict[key] = cls.structTypeDict[key](inputDict[key], ctxt)

        return castedDict

    @classmethod
    def checkValue(cls, value: Union[str, Dict[str, BaseType]], ctxt: Optional[_NetworkContext] = None):

        if isinstance(value, str):
            value = ctxt.lookup(value).structDict.value

        if not hasattr(value, "keys"):
            return False

        if set(value.keys()) != set(cls.structTypeDict.keys()):
            return False

        for key, _value in value.items():
            if not cls.structTypeDict[key].checkPromotion(_value, ctxt):
                return False

        return True

    @classmethod
    def checkPromotion(cls, _other: Union[str, Dict[str, BaseType], Struct], ctxt: Optional[_NetworkContext] = None):

        if isinstance(_other, Struct):
            other = _other.value
        else:
            other = _other

        return cls.checkValue(other, ctxt)

    def __init__(self, structDict: Union[str, Struct, Dict[str, BaseType]], ctxt: Optional[_NetworkContext] = None):
        """Initialize a new struct object

        Parameters
        ----------
        structDict : Union[str, Struct, Dict[str, BaseType]]
            Either an initialized Deeploy-type struct, a string name
            refering to an intialized struct registered in the
            NetworkContext, or a full definition of the struct
            to-be-initialized
        ctxt : Optional[_NetworkContext]
            Current NetworkContext

        Raises
        ------
        Exception
            Raises an Exception if structDict cannot be assigned to a
            struct of layout structTypeDict

        """

        if not self.checkPromotion(structDict, ctxt):
            raise Exception(f"Can't assign {structDict} to {type(self)}!")

        self.value = self._castDict(
            structDict, ctxt
        )  #: structTypeDict: the value of the struct; corresponds to an element with type layout defined in cls.structTypeDict

    def __eq__(self, other):

        if not (hasattr(other, 'typeWidth') and hasattr(other, 'typeName') and hasattr(other, "value")):
            return False

        if any([not key in other.value.keys() for key in self.value.keys()]):
            return False

        return all([self.value[key] == other.value[key] for key in self.value.keys()])

    def __repr__(self):
        _repr = "{"
        pairs = []
        for key, value in self.value.items():
            pairs.append(f".{key} = {str(value)}")
        _repr += (", ").join(pairs)
        _repr += "}"
        return _repr

    def _typeDefRepr(self):
        _repr = "{"
        pairs = []
        for key, value in self.value.items():
            pairs.append(f"{value.typeName} {key}")
        _repr += ("; ").join(pairs)
        _repr += ";}"
        return _repr


def StructClass(typeName: str, _structTypeDict: Dict[str, Type[BaseType]]) -> Type[Struct]:  # type: ignore
    """Helper function to dynamically generate a Struct class from a structTypeDict definition. Used in Closure Generation to capture a closure's arguments.

    Parameters
    ----------
    typeName : str
        Name of the Struct class that is being created
    _structTypeDict : Dict[str, Type[BaseType]]
        Layout of the Struct class that is being created

    Returns
    -------
    Type[Struct]:
        Returns the class definition of a Struct class corresponding
        to the function arguments

    """

    if typeName not in globals().keys():
        retCls = type(typeName, (Struct,), {
            "typeName": typeName,
            "structTypeDict": _structTypeDict,
        })
        globals()[typeName] = retCls
    else:
        retCls = globals()[typeName]

    return retCls


def PointerClass(DeeployType: _DeeployType) -> Type[Pointer[BaseType]]:  # type: ignore
    """Generates a Pointer class definition at runtime that wraps around the given referenceType

    Parameters
    ----------
    DeeployType : _DeeployType
        Type of the underlying referencedType

    Returns
    -------
    Type[Pointer[BaseType]]:
        Returns a unique Pointer class corresponding to a Pointer to
        DeeployType

    """

    typeName = DeeployType.typeName + "Ptr"
    if typeName not in globals().keys():
        retCls = type(typeName, (Pointer,), {"typeWidth": 32, "referencedType": DeeployType})
        globals()[typeName] = retCls
    else:
        retCls = globals()[typeName]

    return retCls
