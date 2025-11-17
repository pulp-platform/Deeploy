# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, PerTensorWaitingStrategy, DmaDirection, Future


class ClDmaFuture(Future):
    """
    Future implementation for GAP9's cl_dma.h API.
    Uses pi_cl_dma_cmd_t structure to track DMA transfers.
    """

    _initTemplate = NodeTemplate("pi_cl_dma_cmd_t ${name};")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("")  # SDK handles allocation automatically

    _waitTemplate = NodeTemplate("pi_cl_dma_cmd_wait(&${name});")


class ClDma(AsyncDma):
    """
    GAP9 Cluster DMA implementation using cl_dma.h high-level API.
    
    This uses the PMSIS standard API (pi_cl_dma_cmd) instead of low-level MCHAN.
    Benefits:
    - Higher abstraction level
    - Better portability
    - Automatic resource management
    - Cleaner API
    """

    _transferTemplates = {
        1: NodeTemplate("pi_cl_dma_cmd(${ext}, ${loc}, ${size}, ${dir}, &${future});"),
        2: NodeTemplate("pi_cl_dma_cmd_2d(${ext}, ${loc}, ${size}, ${stride}, ${length}, ${dir}, &${future});"),
    }
    _waitingStrategy = PerTensorWaitingStrategy(ClDmaFuture)

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)

        transferRank = len(shape)
        # GAP9 cl_dma requires contiguous transfers for innermost dimension
        assert strideExt[
            -1] == 1, "GAP9 cl_dma supports only contiguous transfers of the innermost dimension for external memory"
        if transferRank == 1:
            assert strideLoc[0] == 1, "GAP9 cl_dma supports only contiguous transfers for local memory"
        else:
            assert strideLoc[0] == shape[1] and strideLoc[
                1] == 1, "GAP9 cl_dma supports only contiguous transfers for local memory"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)

        transferRank = len(shape)

        # Use cl_dma API direction enum: PI_CL_DMA_DIR_EXT2LOC (1) or PI_CL_DMA_DIR_LOC2EXT (0)
        operatorRepresentation["dir"] = "PI_CL_DMA_DIR_EXT2LOC" if direction == "ExternalToLocal" else "PI_CL_DMA_DIR_LOC2EXT"
        
        # Total transfer size in bytes (shape already contains byte counts from the framework)
        transferSize = math.prod(shape)
        operatorRepresentation["size"] = transferSize

        # For 2D transfers, add stride and length parameters (already in bytes)
        if transferRank == 2:
            # stride: bytes to add to go to next line (row stride in external memory)
            operatorRepresentation["stride"] = strideExt[0]
            # length: bytes per line (number of bytes after which DMA switches to next line)
            operatorRepresentation["length"] = shape[1]

        return operatorRepresentation
