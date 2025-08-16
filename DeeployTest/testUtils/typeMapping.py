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
from typing import List, Optional

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, int8_t

offsetType = namedtuple("offsetType", ("type", "offset"))

_ALL_DTYPES = {t.typeName: t for t in (*IntegerDataTypes, *FloatDataTypes)}


def parseDataType(name: str):
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
        raise ValueError(f"Unknown data type: {name}")
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


def inferInputType(_input: np.ndarray,
                   signProp: Optional[bool] = None,
                   defaultType = PointerClass(int8_t),
                   defaultOffset = 0,
                   *,
                   autoInfer: bool = True) -> List[offsetType]:

    # WIESEP: We cannot do type inference for empty arrays.
    if np.prod(_input.shape) == 0:
        print(f"Warning: Empty input array for type inference for {_input}!")
        return [(defaultType, defaultOffset)]

    # If the caller provided a manual override, skip all inference.
    if not autoInfer:
        rawType = defaultType.referencedType
        vals = (_input.astype(np.int64) - defaultOffset)
        if not rawType.checkPromotion(vals):
            lo, hi = rawType.typeMin, rawType.typeMax
            raise RuntimeError(f"Provided type {rawType.typeName} with offset {defaultOffset} "
                               f"does not match input values in range [{vals.min()}, {vals.max()}] "
                               f"(expected range [{lo}, {hi}])")

        smallest = rawType
        for caand in sorted(IntegerDataTypes, key = lambda x: x.typeWidth):
            if caand.checkPromotion(vals):
                smallest = caand
                break
        if smallest is not rawType:
            print(f"WARNING: Data spans [{int(vals.min())}, {int(vals.max())}], "
                  f"which would fit in {smallest.typeName}, "
                  f"but user forced {rawType.typeName}.")
        return [(defaultType, defaultOffset)]

    if signProp is None:
        signProp = False

    signedPlatformTypes = [_type for _type in IntegerDataTypes if _type.typeMin < 0]

    matchingTypes = []

    # FIXME: this is okay for now (3 distinctions are fine), but there is implicit
    # knowledge encoded in the order of the checks (i.e. first unsigned, signed
    # and then float). It might be good to extract that implicit knowledge into an ordered list.
    if signProp and isUnsigned(_input) and isInteger(_input):
        for _type in sorted(signedPlatformTypes, key = lambda x: x.typeWidth):
            signPropOffset = (2**(_type.typeWidth - 1))
            if _type.checkPromotion(_input - signPropOffset):
                matchingTypes.append(offsetType(PointerClass(_type), signPropOffset))
    elif isInteger(_input):
        sorted_types = sorted(
            IntegerDataTypes,
            key = lambda t: (t.typeWidth, t.typeMin < 0),
        )

        matchingTypes = []
        for _type in sorted_types:
            if _type.checkPromotion(_input):
                matchingTypes.append(offsetType(PointerClass(_type), 0))
    else:
        for _type in sorted(FloatDataTypes, key = lambda x: x.typeWidth):
            if _type.checkPromotion(_input):
                matchingTypes.append(offsetType(PointerClass(_type), 0))

    if matchingTypes == []:
        raise Exception("Could not find a matching type!")

    return matchingTypes
