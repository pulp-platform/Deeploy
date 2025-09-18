# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
