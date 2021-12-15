# ----------------------------------------------------------------------
#
# File: PULPDataTypes.py
#
# Last edited: 01.06.2023
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

from dataclasses import dataclass
from functools import partial

from Deeploy.AbstractDataTypes import PointerClass, Struct, VoidType
from Deeploy.CommonExtensions.DataTypes import int32_t, uint8_t, uint16_t, uint32_t
from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.FutureExtension.Future import FutureClass

_DMAResolveTemplate = NodeTemplate("""
// PULP CLUSTER DMA Resolve
dory_dma_barrier(&${stateReference});
""")

_DMADispatchTemplate = NodeTemplate("""
// PULP CLUSTER DMA Dispatch
// No dispatch necessary
""")


class DMA_copy(Struct):
    typeName = "DMA_copy"
    structTypeDict = {
        "ext": PointerClass(VoidType),
        "loc": PointerClass(VoidType),
        "hwc_to_chw": uint16_t,
        "stride_2d": uint16_t,
        "number_of_2d_copies": uint16_t,
        "stride_1d": uint16_t,
        "number_of_1d_copies": uint16_t,
        "length_1d_copy": uint16_t,
        "mchan_cmd": uint32_t,
        "dir": int32_t,
        "tid": int32_t
    }


class pi_cl_ram_req_t(Struct):
    typeName = "pi_cl_ram_req_t"
    structTypeDict = {
        "addr": PointerClass(VoidType),
        "pi_ram_addr": PointerClass(VoidType),
        "size": uint32_t,
        "stride": uint32_t,
        "length": uint32_t,
        "is_2d": uint8_t,
        "ext2loc": uint8_t,
    }


@dataclass
class PULPStructDataTypes():
    DMA_copy = DMA_copy
    pi_cl_ram_req_t = pi_cl_ram_req_t


PULPDMAFuture = partial(FutureClass,
                        stateReferenceType = PULPStructDataTypes.DMA_copy,
                        resolveCheckTemplate = _DMAResolveTemplate,
                        dispatchCheckTemplate = _DMADispatchTemplate)
