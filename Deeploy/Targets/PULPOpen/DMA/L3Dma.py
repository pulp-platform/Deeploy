# ----------------------------------------------------------------------
#
# File: L3Dma.py
#
# Last edited: 11.09.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Luka Macan, University of Bologna
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

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, BlockingDmaFromAsyncDmaAdapter, DmaDirection, Future, \
    PerTensorWaitingStrategy


class L3DmaFuture(Future):

    _initTemplate = NodeTemplate("pi_cl_ram_req_t ${name};")
    _deinitTemplate = NodeTemplate("")
    _waitTemplate = NodeTemplate("pi_cl_ram_copy_wait(&${name});")


class L3Dma(AsyncDma):

    _transferTemplates = {
        2:
            NodeTemplate(
                "pi_cl_ram_copy_2d(get_ram_ptr(), ${ext}, ${loc}, ${transfer_size}, ${stride}, ${length}, ${ext2loc}, &${future});"
            )
    }
    _waitingStrategy = PerTensorWaitingStrategy(L3DmaFuture)

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        assert strideExt[-1] == 1, \
            "Mchan supports only contigous transfers of the innermost dimension for external memory"
        assert strideLoc[0] == shape[1] and strideLoc[1] == 1, \
            f"Mchan supports only contigous transfers for local memory. Received local shape: {shape}, stride: {strideLoc}"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)
        operatorRepresentation.update({
            "ext2loc": 1 if direction == "ExternalToLocal" else 0,
            "transfer_size": math.prod(shape),
            "length": shape[1],
            "stride": strideExt[0],
        })
        return operatorRepresentation


# LMACAN: It's a hack because the driver is now working correctly
l3DmaHack = BlockingDmaFromAsyncDmaAdapter(L3Dma())
