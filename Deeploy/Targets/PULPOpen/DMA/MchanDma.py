# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import CodeSnippet, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DirectionWaitingStrategy, DmaDirection, Future


class MchanChannelFuture(Future):

    _initTemplate = NodeTemplate("uint32_t ${name} = (uint32_t) -1;")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("${name} = mchan_channel_alloc();")

    _waitTemplate = NodeTemplate("""
if (${name} <= MCHAN_CHANNEL_ID_MAX) {
    mchan_channel_wait(${name});
    mchan_channel_free(${name});
}
""")


class MchanDma(AsyncDma):

    _transferTemplates = {
        1: NodeTemplate("mchan_transfer_1d(${cmd}, ${loc}, ${ext});"),
        2: NodeTemplate("mchan_transfer_2d_ext_strided(${cmd}, ${loc}, ${ext}, ${size_1d}, ${stride_2d});"),
    }
    _waitingStrategy = DirectionWaitingStrategy(MchanChannelFuture, "channel")

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

    _MAX_1D_TRANSFER_BYTES = 1 << 17  # 131072 bytes: max representable in 17-bit mchan cmd size field

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)

        transferRank = len(shape)

        mchanFlags = 0
        mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0  # direction
        mchanFlags += (1 << 1)  # increment addresses
        mchanFlags += (1 << 2) if transferRank == 2 else 0  # 2d transfer
        mchanFlags += (1 << 3)  # event enable

        mchanTransferSize = math.prod(shape)
        assert mchanTransferSize <= self._MAX_1D_TRANSFER_BYTES, (
            "The transfer size is not representable with 17 bits. "
            f"Received transfer size {mchanTransferSize} that requires "
            f"{math.ceil(math.log2(mchanTransferSize))} bits")

        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize

        if transferRank == 2:
            operatorRepresentation["size_1d"] = shape[1]
            operatorRepresentation["stride_2d"] = strideExt[0]

        return operatorRepresentation

    def transfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                 shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                 direction: DmaDirection, future: Future) -> List[CodeSnippet]:
        # For 1D transfers that exceed the 17-bit mchan limit, split into chunks.
        totalSize = math.prod(shape)
        if len(shape) == 1 and totalSize > self._MAX_1D_TRANSFER_BYTES:
            mchanFlags = 0
            mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0
            mchanFlags += (1 << 1)  # increment addresses
            mchanFlags += (1 << 3)  # event enable
            template = self._transferTemplates[1]
            chunks: List[CodeSnippet] = []
            offset = 0
            while offset < totalSize:
                chunkSize = min(self._MAX_1D_TRANSFER_BYTES, totalSize - offset)
                cmd = (mchanFlags << 17) + chunkSize
                opRepr: OperatorRepresentation = {
                    "loc": f"((char*){localBuffer.name} + {offset})",
                    "ext": f"((char*){externalBuffer.name} + {offset})",
                    "future": future.name,
                    "cmd": cmd,
                }
                chunks.append(CodeSnippet(template, opRepr))
                offset += chunkSize
            return chunks
        return super().transfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)
