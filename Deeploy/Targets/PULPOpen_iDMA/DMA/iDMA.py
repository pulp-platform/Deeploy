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
        1: NodeTemplate(" pulp_idma_transfer_1d_and_wait(${direction}, ${ext}, ${loc}, ${size_1d}); ")
        # 2: NodeTemplate("mchan_transfer_2d_ext_strided(${cmd}, ${loc}, ${ext}, ${size_1d}, ${stride_2d});"),
    }
    _waitingStrategy = DirectionWaitingStrategy(iDMAChannelFuture, "channel_id")

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)

        transferRank = len(shape)
        assert strideExt[
            -1] == 1, "Mchan supports only contigous transfers of the innermost dimension for external memory"
        if transferRank == 1:
            assert strideLoc[0] == 1, "Mchan supports only contigous transfers for local memory"
        else:
            assert strideLoc[0] == shape[1] and strideLoc[
                1] == 1, "Mchan supports only contigous transfers for local memory"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)

        transferRank = len(shape)
        operatorRepresentation["direction"] = 1 if direction == "ExternalToLocal" else 0
        mchanFlags = 0
        mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0  # direction
        mchanFlags += (1 << 1)  # increment addresses
        mchanFlags += (1 << 2) if transferRank == 2 else 0  # 2d transfer
        mchanFlags += (1 << 3)  # event enable

        mchanTransferSize = math.prod(shape)
        assert mchanTransferSize <= 2**17, (
            "The Dma transfer size for mchan should be representable with 17 bits, "
            f"current number of bits required is {math.ceil(math.log2(mchanTransferSize))}")

        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize
        operatorRepresentation["size_1d"] = shape[0]
        # if transferRank == 2:
        #     operatorRepresentation["size_1d"] = shape[1]
        #     operatorRepresentation["stride_2d"] = strideExt[0]

        return operatorRepresentation