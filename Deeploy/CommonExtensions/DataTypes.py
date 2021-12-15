# ----------------------------------------------------------------------
#
# File: BasicDataTypes.py
#
# Last edited: 31.08.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

from typing import Tuple, Type

from Deeploy.AbstractDataTypes import IntegerImmediate


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


SignedIntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = (int8_t, int16_t, int32_t, int64_t)
UnsignedIntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = (uint8_t, uint16_t, uint32_t, uint64_t)
IntegerDataTypes: Tuple[Type[IntegerImmediate], ...] = (sorted((
    *SignedIntegerDataTypes,
    *UnsignedIntegerDataTypes,
),
                                                               key = lambda _type: _type.typeWidth))
