# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
        "length_1d_copy": uint32_t,
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
