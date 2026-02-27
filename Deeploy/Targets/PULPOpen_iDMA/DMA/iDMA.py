# Copyright (c) 2025 FondazioneChipsIT
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DirectionWaitingStrategy, DmaDirection, Future


class iDMAChannelFuture(Future):

    _initTemplate = NodeTemplate("")
    _deinitTemplate = NodeTemplate("")
    _waitTemplate = NodeTemplate("")
    _allocTemplate = NodeTemplate("")


class iDMA(AsyncDma):

    _transferTemplates = {
        1:
            NodeTemplate("pulp_idma_transfer_1d_and_wait(${direction}, ${ext}, ${loc}, ${length}); "),
        2:
            NodeTemplate(
                "pulp_idma_transfer_2d_and_wait(${direction}, ${ext}, ${loc}, ${length}, ${strideExt}, ${strideLoc}, ${num_reps});"
            )
    }
    _waitingStrategy = DirectionWaitingStrategy(iDMAChannelFuture, "channel_id")

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)

        # Here are some assertions on the iDMA transfer parameters
        transferRank = len(shape)

        assert transferRank == 1 or transferRank == 2, "Only 1D and 2D transfers are supported for now"

        if transferRank == 1:
            length = shape[0]
        elif transferRank == 2:
            length = shape[1]

        iDMA_transfer_size = math.prod(shape)

        assert iDMA_transfer_size <= 2**16, (
            "iDMA transfer size should be representable with 16 bits, "
            f"current number of bits required is {math.ceil(math.log2(iDMA_transfer_size))}")

        if transferRank == 2:
            assert strideExt[0] >= length, "External stride must be at least equal to the length"
            assert strideLoc[0] >= length, "Local stride must be at least equal to the length"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)

        transferRank = len(shape)
        operatorRepresentation["direction"] = 1 if direction == "ExternalToLocal" else 0

        iDMA_transfer_size = math.prod(shape)

        if transferRank == 1:
            operatorRepresentation["length"] = shape[0]
        elif transferRank == 2:
            operatorRepresentation["length"] = shape[1]
            operatorRepresentation["strideExt"] = strideExt[0]
            operatorRepresentation["strideLoc"] = strideLoc[0]
            operatorRepresentation["num_reps"] = iDMA_transfer_size // shape[1]

        return operatorRepresentation
