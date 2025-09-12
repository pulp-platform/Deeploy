# ----------------------------------------------------------------------
#
# File: SnitchDma.py
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

from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DmaDirection, Future, TensorGroupWaitingStrategy


class SnitchBarrierFuture(Future):
    _initTemplate = NodeTemplate("")
    _deinitTemplate = NodeTemplate("")
    _waitTemplate = NodeTemplate("if (snrt_is_dm_core()) snrt_dma_wait_all();")


# LMACAN: TODO: Add single transfer waiting
class SnitchFuture(Future):
    _initTemplate = NodeTemplate("uint16_t ${name};")
    _deinitTemplate = NodeTemplate("")
    _waitTemplate = NodeTemplate("if (snrt_is_dm_core()) snrt_dma_wait(${name});")


class SnitchDma(AsyncDma):

    _transferTemplates = {
        2:
            NodeTemplate(
                "if (snrt_is_dm_core()) snrt_dma_start_2d(${dest}, ${src}, ${size}, ${stride_dest}, ${stride_src}, ${repeat});"
            ),
    }
    _waitingStrategy = TensorGroupWaitingStrategy(SnitchBarrierFuture, "")

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        assert strideLoc[1] == 1 and strideExt[1] == 1, f"Supports only contigous transfers in the innermost dimension"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        _ = future
        operatorRepresentation: OperatorRepresentation = {
            "dest": localBuffer.name if direction == "ExternalToLocal" else externalBuffer.name,
            "src": externalBuffer.name if direction == "ExternalToLocal" else localBuffer.name,
            "repeat": shape[0],
            "size": shape[1],
            "stride_dest": strideLoc[0] if direction == "ExternalToLocal" else strideExt[0],
            "stride_src": strideExt[0] if direction == "ExternalToLocal" else strideLoc[0],
        }
        return operatorRepresentation
