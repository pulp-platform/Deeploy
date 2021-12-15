# ----------------------------------------------------------------------
#
# File: MemPoolDataTypes.py
#
# Last edited: 08.01.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
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

from dataclasses import dataclass

from Deeploy.AbstractDataTypes import PointerClass, Struct
from Deeploy.CommonExtensions.DataTypes import int32_t, uint8_t


class ita_quant_t(Struct):
    typeName = "ita_quant_t"
    structTypeDict = {
        'eps_mult': PointerClass(uint8_t),
        'right_shift': PointerClass(uint8_t),
        'add': PointerClass(int32_t)
    }


@dataclass
class MemPoolStructDataTypes():
    ita_quant_t = ita_quant_t
