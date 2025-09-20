# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DirectionWaitingStrategy, DmaDirection, Future


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
    _waitingStrategy = DirectionWaitingStrategy(SnitchBarrierFuture, "")

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
