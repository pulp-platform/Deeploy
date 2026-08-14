# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, BarrierWaitingStrategy, DmaDirection, Future


class SnitchBarrierFuture(Future):
    _initTemplate = NodeTemplate("")
    _deinitTemplate = NodeTemplate("")
    _allocTemplate = NodeTemplate("")
    _waitTemplate = NodeTemplate("if (snrt_is_dm_core()) snrt_dma_wait_all();")


# LMACAN: TODO: Add single transfer waiting
class SnitchFuture(Future):
    _initTemplate = NodeTemplate("snrt_dma_txid_t ${name} = (snrt_dma_txid_t) -1;")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("")

    _waitTemplate = NodeTemplate(
        "if ( (${name} != ( (snrt_dma_txid_t) -1) ) && snrt_is_dm_core() ) snrt_dma_wait(${name});")


class SnitchDma(AsyncDma):

    _transferTemplates = {
        2:
            NodeTemplate("""
            if (snrt_is_dm_core()) {
                snrt_dma_start_2d(${dest}, ${src}, ${size}, ${stride_dest}, ${stride_src}, ${repeat});
            }
            """),
    }
    # Wait for all outstanding transfers rather than for an individual one.
    #
    # snrt_dma_wait compares against completed_id, which idma advances when the
    # ND midend sees burst_rsp.last. That is upstream of the write datapath, so
    # it can move before the transferred data is visible, and the barrier that
    # follows then releases the compute cores onto a tile the DMA has not
    # finished writing. snrt_dma_wait_all instead polls the busy flag, which
    # covers every pipeline stage.
    #
    # Two further details make per-transfer waiting unusable here: dmcpyi
    # returns the same ID for transfers issued back to back.
    #
    # This also drops the self-copy that used to follow every transfer: it
    # existed to bump completed_id past the last transaction ID, which the
    # strictly-greater comparison in the previously pinned runtime required.
    _waitingStrategy = BarrierWaitingStrategy(SnitchBarrierFuture, "dma_barrier")

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
        operatorRepresentation: OperatorRepresentation = {
            "dest": localBuffer.name if direction == "ExternalToLocal" else externalBuffer.name,
            "src": externalBuffer.name if direction == "ExternalToLocal" else localBuffer.name,
            "repeat": shape[0],
            "size": shape[1],
            "stride_dest": strideLoc[0] if direction == "ExternalToLocal" else strideExt[0],
            "stride_src": strideExt[0] if direction == "ExternalToLocal" else strideLoc[0],
            "future": future.name
        }
        return operatorRepresentation
