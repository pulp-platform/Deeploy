# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DmaDirection, Future, PerTensorWaitingStrategy


class MchanChannelFuture(Future):

    _initTemplate = NodeTemplate("""
    % if comment:
    // ${comment}
    % endif
    uint32_t ${name} = 0;
    """)

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("""
     % if comment:
    // ${comment}
    % endif
    ${name} = mchan_channel_alloc();
    """)

    _waitTemplate = NodeTemplate("""
    % if comment:
    // ${comment}
    % endif
    mchan_channel_wait(${name});
    mchan_channel_free(${name});
    """)


class MchanDma(AsyncDma):

    _transferTemplates = {
        1:
            NodeTemplate("""
        % if comment:
        // ${comment}
        % endif
        mchan_transfer_1d(${cmd}, ${loc}, ${ext});
        """),
        2:
            NodeTemplate("""
        % if comment:
        // ${comment}
        % endif
        mchan_transfer_2d_ext_strided(${cmd}, ${loc}, ${ext}, ${size_1d}, ${stride_2d});
        """),
    }
    _waitingStrategy = PerTensorWaitingStrategy(MchanChannelFuture)

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

    def transferOpRepr(self,
                       externalBuffer: VariableBuffer,
                       localBuffer: VariableBuffer,
                       shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...],
                       strideLoc: Tuple[int, ...],
                       direction: DmaDirection,
                       future: Future,
                       comment: str = "") -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future, comment)

        transferRank = len(shape)

        mchanFlags = 0
        mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0  # direction
        mchanFlags += (1 << 1)  # increment addresses
        mchanFlags += (1 << 2) if transferRank == 2 else 0  # 2d transfer
        mchanFlags += (1 << 3)  # event enable

        mchanTransferSize = math.prod(shape)
        mchanTransferSizeBits = math.ceil(math.log2(mchanTransferSize))
        assert mchanTransferSizeBits <= 17, (
            "The transfer size is not representable with 17 bits. "
            f"Received transfer size {mchanTransferSize} that requires {mchanTransferSizeBits}")

        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize

        if transferRank == 2:
            operatorRepresentation["size_1d"] = shape[1]
            operatorRepresentation["stride_2d"] = strideExt[0]

        return operatorRepresentation
