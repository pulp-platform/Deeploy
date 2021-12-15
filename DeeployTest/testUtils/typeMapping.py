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
from Deeploy.CommonExtensions.DataTypes import IntegerDataTypes, int8_t

offsetType = namedtuple("offsetType", ("type", "offset"))


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
                   defaultOffset = 0) -> List[offsetType]:

    # WIESEP: We cannot do type inference for empty arrays.
    if np.prod(_input.shape) == 0:
        print(f"Warning: Empty input array for type inference for {_input}!")
        return [(defaultType, defaultOffset)]

    if not isInteger(_input):
        raise Exception("Deeploy currently only handles integer types!")

    if signProp is None:
        signProp = False

    signedPlatformTypes = [_type for _type in IntegerDataTypes if _type.typeMin < 0]

    matchingTypes = []

    if signProp and isUnsigned(_input):
        for _type in sorted(signedPlatformTypes, key = lambda x: x.typeWidth):
            signPropOffset = (2**(_type.typeWidth - 1))
            if _type.checkPromotion(_input - signPropOffset):
                matchingTypes.append(offsetType(PointerClass(_type), signPropOffset))
    else:
        for _type in sorted(IntegerDataTypes, key = lambda x: x.typeWidth):
            if _type.checkPromotion(_input):
                matchingTypes.append(offsetType(PointerClass(_type), 0))

    if matchingTypes == []:
        raise Exception("Could not find a matching type!")

    return matchingTypes
