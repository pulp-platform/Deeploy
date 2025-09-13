# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Tuple, Type, Union

import numpy.typing as npt

from Deeploy.AbstractDataTypes import FloatImmediate, IntegerImmediate


class int8_t(IntegerImmediate):
    typeName = "int8_t"
    typeWidth = 8
    signed = True


class int16_t(IntegerImmediate):
    typeName = "int16_t"
    typeWidth = 16
    signed = True


class int32_t(IntegerImmediate):
    typeName = "int32_t"
    typeWidth = 32
    signed = True


class int64_t(IntegerImmediate):
    typeName = "int64_t"
    typeWidth = 64
    signed = True


class uint8_t(IntegerImmediate):
    typeName = "uint8_t"
    typeWidth = 8
    signed = False


class uint16_t(IntegerImmediate):
    typeName = "uint16_t"
    typeWidth = 16
    signed = False


class uint32_t(IntegerImmediate):
    typeName = "uint32_t"
    typeWidth = 32
    signed = False


class uint64_t(IntegerImmediate):
    typeName = "uint64_t"
    typeWidth = 64
    signed = False


class bfloat16_t(FloatImmediate):
    typeName = "bfloat16_t"
    typeWidth = 16
    typeMantissa = 7
    typeExponent = 8


class float16_t(FloatImmediate):
    typeName = "float16_t"
    typeWidth = 16
    typeMantissa = 10
    typeExponent = 5


class float32_t(FloatImmediate):
    typeName = "float32_t"
    typeWidth = 32
    typeMantissa = 23
    typeExponent = 8


class float64_t(FloatImmediate):
    typeName = "float64_t"
    typeWidth = 64
    typeMantissa = 52
    typeExponent = 11


SignedIntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = (int8_t, int16_t, int32_t, int64_t)
UnsignedIntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = (uint8_t, uint16_t, uint32_t, uint64_t)
IntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = (sorted((
    *SignedIntegerDataTypes,
    *UnsignedIntegerDataTypes,
),
                                                               key = lambda _type: _type.typeWidth))
FloatDataTypes: Tuple[Type[FloatImmediate], ...] = (bfloat16_t, float16_t, float32_t, float64_t)


def minimalIntegerType(value: Union[int, Iterable[int], npt.NDArray]) -> Type[IntegerImmediate]:
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
    # Sort data types by typeWidth
    sorted_types = sorted(
        FloatDataTypes,
        key = lambda t: t.typeWidth,
    )

    for _type in sorted_types:
        if _type.checkValue(value):
            return _type

    raise RuntimeError(f"Couldn't find appropriate float type for value: {value}")
