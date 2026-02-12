# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Tuple, Type, Union

import numpy.typing as npt

from Deeploy.AbstractDataTypes import FloatImmediate, IntegerImmediate


class int8_t(IntegerImmediate):
    """8-bit signed integer type."""
    typeName = "int8_t"
    typeWidth = 8
    signed = True


class int16_t(IntegerImmediate):
    """16-bit signed integer type."""
    typeName = "int16_t"
    typeWidth = 16
    signed = True


class int32_t(IntegerImmediate):
    """32-bit signed integer type."""
    typeName = "int32_t"
    typeWidth = 32
    signed = True


class int64_t(IntegerImmediate):
    """64-bit signed integer type."""
    typeName = "int64_t"
    typeWidth = 64
    signed = True


class uint8_t(IntegerImmediate):
    """8-bit unsigned integer type."""
    typeName = "uint8_t"
    typeWidth = 8
    signed = False


class uint16_t(IntegerImmediate):
    """16-bit unsigned integer type."""
    typeName = "uint16_t"
    typeWidth = 16
    signed = False


class uint32_t(IntegerImmediate):
    """32-bit unsigned integer type."""
    typeName = "uint32_t"
    typeWidth = 32
    signed = False


class uint64_t(IntegerImmediate):
    """64-bit unsigned integer type."""
    typeName = "uint64_t"
    typeWidth = 64
    signed = False


class bfloat16_t(FloatImmediate):
    """16-bit bfloat float type with 7-bit mantissa and 8-bit exponent."""
    typeName = "bfloat16_t"
    typeWidth = 16
    typeMantissa = 7
    typeExponent = 8


class float16_t(FloatImmediate):
    """16-bit float type with 10-bit mantissa and 5-bit exponent."""
    typeName = "float16_t"
    typeWidth = 16
    typeMantissa = 10
    typeExponent = 5


class float32_t(FloatImmediate):
    """32-bit float type with 23-bit mantissa and 8-bit exponent."""
    typeName = "float32_t"
    typeWidth = 32
    typeMantissa = 23
    typeExponent = 8


class float64_t(FloatImmediate):
    """64-bit float type with 11-bit mantissa and 52-bit exponent."""
    typeName = "float64_t"
    typeWidth = 64
    typeMantissa = 52
    typeExponent = 11


SignedIntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = (int8_t, int16_t, int32_t, int64_t)
UnsignedIntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = (uint8_t, uint16_t, uint32_t, uint64_t)
IntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = tuple(
    sorted((
        *SignedIntegerDataTypes,
        *UnsignedIntegerDataTypes,
    ), key = lambda _type: _type.typeWidth))
FloatDataTypes: Tuple[Type[FloatImmediate], ...] = (bfloat16_t, float16_t, float32_t, float64_t)


def minimalIntegerType(value: Union[int, Iterable[int], npt.NDArray]) -> Type[IntegerImmediate]:
    """Returns the minimal integer type that can represent all values in the given list.

    Parameters
    ----------
    value : Union[int, Iterable[int], npt.NDArray]
        The list of integer values to analyze.

    Returns
    -------
    Type[IntegerImmediate]
        The minimal integer type that can represent all values.
    """
    # Sort data types by typeWidth and signedness (unsigned types go first)
    sorted_types = sorted(
        IntegerDataTypes,
        key = lambda t: (t.typeWidth, t.typeMin < 0),
    )

    for _type in sorted_types:
        if _type.checkValue(value):
            return _type

    raise RuntimeError(f"Couldn't find appropriate integer type for value: {value}")


def minimalFloatType(value: Union[float, Iterable[float], npt.NDArray]) -> Type[FloatImmediate]:
    """Returns the minimal float type that can represent all values in the given list.

    Parameters
    ----------
    value : Union[float, Iterable[float], npt.NDArray]
        The list of float values to analyze.

    Returns
    -------
    Type[FloatImmediate]
        The minimal float type that can represent all values.
    """
    # Sort data types by typeWidth
    sorted_types = sorted(
        FloatDataTypes,
        key = lambda t: t.typeWidth,
    )

    for _type in sorted_types:
        if _type.checkValue(value):
            return _type

    raise RuntimeError(f"Couldn't find appropriate float type for value: {value}")
