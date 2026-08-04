# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Type

import numpy as np
import numpy.typing as npt

from Deeploy.AbstractDataTypes import BaseType, IntegerImmediate, Pointer, PointerClass
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, float32_t, int8_t, int16_t, int32_t, \
    minimalFloatType, minimalIntegerType, uint8_t, uint16_t, uint32_t

_ALL_DTYPES: dict[str, type] = {t.typeName: t for t in (*IntegerDataTypes, *FloatDataTypes)}


def parseDataType(name: str) -> type:
    """Parses a data type from its name.

    Parameters
    ----------
    name : str
        The name of the data type.

    Returns
    -------
    class
        The corresponding data type class.

    Raises
    ------
    ValueError
        If the provided data type name is unknown.
    """
    if name not in _ALL_DTYPES:
        allowed = ", ".join(sorted(_ALL_DTYPES))
        raise ValueError(f"Unknown data type: {name}. Allowed: {allowed}")
    return _ALL_DTYPES[name]


def isInteger(x: npt.NDArray) -> bool:
    return np.abs((x.astype(int) - x)).max() <= 0.001


def inferMinimalType(values: np.ndarray,
                     default: Type[BaseType] = int8_t,
                     original_dtype: np.dtype = None) -> Type[BaseType]:
    # WIESEP: We cannot do type inference for empty arrays.
    if np.prod(values.shape) == 0:
        print(f"Warning: Empty input array for type inference for {values}!")
        return default

    # For all-zero arrays, use original dtype to distinguish int vs float
    if np.all(values == 0) and original_dtype is not None:
        if np.issubdtype(original_dtype, np.floating):
            return minimalFloatType(values)
        return minimalIntegerType(values)

    if isInteger(values):
        return minimalIntegerType(values)
    else:
        return minimalFloatType(values)


def signPropTypeAndOffset(_type: Type[IntegerImmediate]) -> Tuple[Type[IntegerImmediate], int]:
    if _type.signed:
        return _type, 0

    unsigned2signed = {
        unsigned.typeName: signed for unsigned, signed in zip([t for t in IntegerDataTypes if t.typeMin == 0
                                                              ], [t for t in IntegerDataTypes if t.typeMin < 0])
    }

    signedType = unsigned2signed[_type.typeName]
    return signedType, 2**(signedType.typeWidth - 1)


def inferTypeAndOffset(values: np.ndarray,
                       signProp: bool = False,
                       original_dtype: np.dtype = None) -> Tuple[Type[Pointer], int]:
    """Infers the data type of the provided input array.

    Parameters
    ----------
    values : np.ndarray
        The input array for which to infer the data type.

    signProp : bool
        Whether to consider signedness when inferring the data type.

    original_dtype : np.dtype, optional
        Original numpy dtype before float64 cast, used to resolve all-zero ambiguity.

    Returns
    -------
    Tuple[Type[BaseType], int]
        The inferred type and offset
    """

    _type = inferMinimalType(values, original_dtype = original_dtype)

    if signProp and issubclass(_type, IntegerImmediate):
        _type, offset = signPropTypeAndOffset(_type)
    else:
        offset = 0

    return PointerClass(_type), offset


def baseTypeFromName(name: str) -> Type[BaseType]:
    if name == "int8_t":
        return int8_t
    elif name == "uint8_t":
        return uint8_t
    elif name == "int16_t":
        return int16_t
    elif name == "uint16_t":
        return uint16_t
    elif name == "int32_t":
        return int32_t
    elif name == "uint32_t":
        return uint32_t
    elif name == "float32_t":
        return float32_t
    else:
        raise RuntimeError(f"Unrecognized name {name}")


def dtypeFromDeeployType(_ty: Type[BaseType]) -> npt.DTypeLike:
    if _ty == int8_t:
        return np.int8
    elif _ty == uint8_t:
        return np.uint8
    elif _ty == int16_t:
        return np.int16
    elif _ty == uint16_t:
        return np.uint16
    elif _ty == int32_t:
        return np.int32
    elif _ty == uint32_t:
        return np.uint32
    elif _ty == float32_t:
        return np.float32
    else:
        raise RuntimeError(f"Unimplemented conversion for type {_ty.typeName}")
