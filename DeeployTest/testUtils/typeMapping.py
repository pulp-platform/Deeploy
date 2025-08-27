# ----------------------------------------------------------------------
#
# File: typeMapping.py
#
# Last edited: 22.05.2023
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

from collections import namedtuple
from typing import List

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, int8_t

offsetType = namedtuple("offsetType", ("type", "offset"))

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


def isInteger(_input: np.array) -> bool:
    if np.abs((_input.astype(int) - _input)).max() > 0.001:
        return False
    return True


def isUnsigned(_input: np.array) -> bool:
    if (_input).min() < 0:
        return False
    return True


def dataWidth(n):
    count = 0
    n = np.abs(int(n - 1))
    while (n > 0):
        count += 1
        n = n >> 8
    ret = 2**(count + 2)
    if ret < 8:
        ret = 8
    return ret


def inferInputType(values: np.ndarray,
                   signProp: bool = False,
                   defaultType = int8_t,
                   defaultOffset = 0) -> List[offsetType]:
    """Infers the data type of the provided input array.

    Parameters
    ----------
    values : np.ndarray
        The input array for which to infer the data type.

    signProp : bool
        Whether to consider signedness when inferring the data type.

    defaultType : type
        The default data type to use if inference fails.

    defaultOffset : int
        The default offset to use if inference fails.

    Returns
    -------
    List[offsetType]
        A list of inferred data types and their corresponding offsets.
    """

    # WIESEP: We cannot do type inference for empty arrays.
    if np.prod(values.shape) == 0:
        print(f"Warning: Empty input array for type inference for {values}!")
        return [(defaultType, defaultOffset)]

    signedPlatformTypes = [_type for _type in IntegerDataTypes if _type.typeMin < 0]

    matchingTypes = []

    # There is implicit knowledge encoded in the order of the checks (i.e. first unsigned, signed
    # and then float).
    if signProp and isUnsigned(values) and isInteger(values):
        for _type in sorted(signedPlatformTypes, key = lambda x: x.typeWidth):
            signPropOffset = (2**(_type.typeWidth - 1))
            if _type.checkPromotion(values - signPropOffset):
                matchingTypes.append(offsetType(PointerClass(_type), signPropOffset))
    elif isInteger(values):
        sorted_types = sorted(
            IntegerDataTypes,
            key = lambda t: (t.typeWidth, t.typeMin < 0),
        )

        for _type in sorted_types:
            if _type.checkPromotion(values):
                matchingTypes.append(offsetType(PointerClass(_type), 0))
    else:
        for _type in sorted(FloatDataTypes, key = lambda x: x.typeWidth):
            if _type.checkPromotion(values):
                matchingTypes.append(offsetType(PointerClass(_type), 0))

    if not matchingTypes:
        raise RuntimeError("Could not find a matching type!")

    return matchingTypes
