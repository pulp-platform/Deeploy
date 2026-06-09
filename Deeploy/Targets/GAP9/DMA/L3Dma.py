# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, BlockingDmaFromAsyncDmaAdapter, DmaDirection, Future, \
    PerTensorWaitingStrategy


class GAP9L3DmaFuture(Future):

    _initTemplate = NodeTemplate("pi_cl_ram_req_t ${name} = {0};")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("")

    _waitTemplate = NodeTemplate("""
    if (${name}.size != 0) {
        pi_cl_ram_copy_wait(&${name});
    }""")


class GAP9L3Dma(AsyncDma):

    # NOTE: The GAP9 OctoSPI 2-D strided DMA (pi_cl_ram_copy_2d) is broken on real
    # silicon for genuine multi-line transfers — it works in GVSoC's functional DMA
    # model but delivers garbage on hardware (NaN/inf/FLT_MAX in conv outputs at
    # --defaultMemLevel=L3). 1-D contiguous copies work correctly. So we decompose
    # every 2-D transfer into a loop of 1-D pi_cl_ram_copy line copies. The future
    # request is reused per line, so each line is waited on before the next reuses it.
    _transferTemplates = {
        2:
            NodeTemplate("""
for (uint32_t _gap9_l3_line = 0; _gap9_l3_line < (${transfer_size}) / (${length}); _gap9_l3_line++) {
    pi_cl_ram_copy(get_ram_ptr(), (uint32_t)${ext} + _gap9_l3_line * (${stride}), (void *)((char *)${loc} + _gap9_l3_line * (${length})), ${length}, ${ext2loc}, &${future});
    pi_cl_ram_copy_wait(&${future});
}""")
    }
    _waitingStrategy = PerTensorWaitingStrategy(GAP9L3DmaFuture)

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        assert strideExt[-1] == 1, \
            "GAP9 RAM API requires contiguous transfers of the innermost dimension for external memory"
        assert strideLoc[0] == shape[1] and strideLoc[1] == 1, \
            f"GAP9 RAM API requires contiguous transfers for local memory. Received local shape: {shape}, stride: {strideLoc}"

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


# Blocking adapter for L3 DMA (used in GAP9 L3 tiling)
gap9L3DmaHack = BlockingDmaFromAsyncDmaAdapter(GAP9L3Dma())
