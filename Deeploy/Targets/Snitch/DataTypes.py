# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.AbstractDataTypes import Struct
from Deeploy.AbstractDataTypes import VoidType
from Deeploy.CommonExtensions.DataTypes import uint16_t


class Snitch_DMA_copy(Struct):
    typeName = "DMA_copy"
    structTypeDict = {
        "dst": PointerClass(VoidType),
        "src": PointerClass(VoidType),
        "size": uint16_t,
        "dst_stride": uint16_t,
        "src_stride": uint16_t,
        "repeat": uint16_t,
        "tid": uint16_t
    }
