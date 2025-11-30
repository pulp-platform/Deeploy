# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DirectionWaitingStrategy, DmaDirection, Future


class MchanTransferFuture(Future):
    """
    Future implementation for GAP9's MCHAN v7 low-level API.
    Based on DORY's implementation: https://github.com/pulp-platform/dory

    Uses direct hardware register access for maximum performance.
    """

    _initTemplate = NodeTemplate("int ${name} = -1;")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("${name} = mchan_transfer_get_id();")

    _waitTemplate = NodeTemplate("""
if (${name} >= 0) {
    mchan_transfer_wait(${name});
    mchan_transfer_free(${name});
}
""")


class GAP9MchanDma(AsyncDma):
    """
    GAP9 Cluster DMA implementation using MCHAN v7 low-level API.

    This implementation follows DORY's approach for GAP9:
    - Direct hardware register access for MCHAN v7
    - Manual transfer ID management
    - Support for 1D, 2D, and 3D transfers
    - Event-based or polled synchronization

    References:
    - DORY GAP9: https://github.com/pulp-platform/dory/tree/master/dory/Hardware_targets/PULP/GAP9
    - MCHAN v7 specification in GAP9 documentation
    """

    _transferTemplates = {
        1: NodeTemplate("{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, .loc = ${loc}, .ext = ${ext} }; mchan_transfer_push_1d(__mchan_tmp); }"),
        2: NodeTemplate("{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, .loc = ${loc}, .ext = ${ext}, .ext_size_1d = ${size_1d}, .ext_stride_1d = ${stride_2d} }; mchan_transfer_push_2d(__mchan_tmp); }"),
    }
    _waitingStrategy = DirectionWaitingStrategy(MchanTransferFuture, "transfer")

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)

        transferRank = len(shape)
        # MCHAN v7 requires contiguous transfers for innermost dimension in external memory
        assert strideExt[
            -1] == 1, "GAP9 MCHAN supports only contiguous transfers of the innermost dimension for external memory"

        # Local memory (TCDM) must also be contiguous
        if transferRank == 1:
            assert strideLoc[0] == 1, "GAP9 MCHAN supports only contiguous transfers for local memory"
        else:
            assert strideLoc[0] == shape[1] and strideLoc[
                1] == 1, "GAP9 MCHAN supports only contiguous transfers for local memory"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)

        transferRank = len(shape)

        # Build MCHAN command using flags from mchan.h
        # We construct the cmd value in Python and let the C code use the macros
        mchanFlags = 0
        mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0  # direction
        mchanFlags += (1 << 1)  # increment addresses
        mchanFlags += (1 << 2) if transferRank == 2 else 0  # 2d transfer
        mchanFlags += (1 << 3)  # event enable

        mchanTransferSize = math.prod(shape)
        mchanTransferSizeBits = math.ceil(math.log2(mchanTransferSize)) if mchanTransferSize > 0 else 0
        assert mchanTransferSizeBits <= 17, (
            "The transfer size is not representable with 17 bits. "
            f"Received transfer size {mchanTransferSize} that requires {mchanTransferSizeBits} bits")

        # cmd = (flags << 17) + size, matching PULPOpen MchanDma pattern
        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize
        operatorRepresentation["size"] = mchanTransferSize

        if transferRank == 2:
            operatorRepresentation["size_1d"] = shape[1]
            operatorRepresentation["stride_2d"] = strideExt[0]

        return operatorRepresentation
