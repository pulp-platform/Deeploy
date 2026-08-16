# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import CodeSnippet, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DirectionWaitingStrategy, DmaDirection, Future


class MchanTransferFuture(Future):
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

    # MCHAN encodes the transfer length in the low MCHAN_TRANSFER_LEN_SIZE (17 on
    # GAP9) bits of cmd, so a single command can move at most 2**17 - 1 bytes.
    # Anything larger is issued as several commands under the same transfer id --
    # the generated code already pushes several descriptors per id and waits once,
    # so this needs no new machinery. Chunk counts are decided at code-generation
    # time, where the shape is known.
    MAX_TRANSFER_SIZE = (1 << 17) - 1

    _transferTemplates = {
        1:
            NodeTemplate(
                "{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, .loc = ${loc}, .ext = ${ext} }; mchan_transfer_push_1d(__mchan_tmp); }"
            ),
        2:
            NodeTemplate(
                "{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, .loc = ${loc}, .ext = ${ext}, .ext_size_1d = ${size_1d}, .ext_stride_1d = ${stride_2d} }; mchan_transfer_push_2d(__mchan_tmp); }"
            ),
    }

    _chunkedTransferTemplates = {
        1:
            NodeTemplate("""
{
  int32_t __mchan_rem = ${size};
  int32_t __mchan_off = 0;
  while (__mchan_rem > 0) {
    int32_t __mchan_n = __mchan_rem > ${chunk} ? ${chunk} : __mchan_rem;
    mchan_transfer_t __mchan_tmp = { .cmd = ${flags_shifted} + __mchan_n, .size = __mchan_n,
                                     .loc = (void *)((char *)(${loc}) + __mchan_off),
                                     .ext = (void *)((char *)(${ext}) + __mchan_off) };
    mchan_transfer_push_1d(__mchan_tmp);
    __mchan_off += __mchan_n;
    __mchan_rem -= __mchan_n;
  }
}
"""),
        2:
            NodeTemplate("""
{
  int32_t __mchan_row = 0;
  while (__mchan_row < ${rows}) {
    int32_t __mchan_k = (${rows} - __mchan_row) > ${chunk_rows} ? ${chunk_rows} : (${rows} - __mchan_row);
    mchan_transfer_t __mchan_tmp = { .cmd = ${flags_shifted} + __mchan_k * ${size_1d},
                                     .size = __mchan_k * ${size_1d},
                                     .loc = (void *)((char *)(${loc}) + __mchan_row * ${size_1d}),
                                     .ext = (void *)((char *)(${ext}) + __mchan_row * ${stride_2d}),
                                     .ext_size_1d = ${size_1d}, .ext_stride_1d = ${stride_2d} };
    mchan_transfer_push_2d(__mchan_tmp);
    __mchan_row += __mchan_k;
  }
}
"""),
    }
    _waitingStrategy = DirectionWaitingStrategy(MchanTransferFuture, "transfer")

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def transfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                 shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                 direction: DmaDirection, future: Future) -> List[CodeSnippet]:
        self.checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        opRepr = self.transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)
        if math.prod(shape) > self.MAX_TRANSFER_SIZE:
            template = self._chunkedTransferTemplates[len(shape)]
        else:
            template = self._transferTemplates[len(shape)]
        return [CodeSnippet(template, opRepr)]

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

        # cmd = (flags << 17) + size, matching PULPOpen MchanDma pattern
        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize
        operatorRepresentation["size"] = mchanTransferSize
        operatorRepresentation["flags_shifted"] = mchanFlags << 17

        if transferRank == 2:
            operatorRepresentation["size_1d"] = shape[1]
            operatorRepresentation["stride_2d"] = strideExt[0]

        if mchanTransferSize > self.MAX_TRANSFER_SIZE:
            # Note the bound is 2**17 - 1, not 2**17: a size of exactly 131072
            # carries into the direction flag and silently reverses the transfer.
            if transferRank == 1:
                operatorRepresentation["chunk"] = self.MAX_TRANSFER_SIZE
            else:
                size1d = shape[1]
                chunkRows = self.MAX_TRANSFER_SIZE // size1d
                assert chunkRows >= 1, (
                    f"A single 2D row of {size1d} B exceeds the {self.MAX_TRANSFER_SIZE} B MCHAN transfer limit; "
                    "the tile must be split further along the innermost dimension")
                operatorRepresentation["rows"] = shape[0]
                operatorRepresentation["chunk_rows"] = chunkRows

        return operatorRepresentation
