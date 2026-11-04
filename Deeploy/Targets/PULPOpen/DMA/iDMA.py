# Copyright (c) 2025 FondazioneChipsIT
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DirectionWaitingStrategy, DmaDirection, Future

# HAL pulp_cl_idma_* calls take the (row) length as unsigned short;
# 1D goes through deeploy_idma_transfer_1d (pulp_idma_utils.h), split at runtime
_MAX_LEN = 2**16 - 1

# Direction encoding expected by the deeploy_idma_transfer_* helpers in
# pulp_idma_utils.h (`direction == 1` means L2 -> L1): keep the two in sync
_DIR_EXT_TO_LOC = 1
_DIR_LOC_TO_EXT = 0


class iDMAChannelFuture(Future):

    _initTemplate = NodeTemplate("")
    _deinitTemplate = NodeTemplate("")
    _waitTemplate = NodeTemplate("deeploy_idma_wait_${name}();")
    _allocTemplate = NodeTemplate("")


class iDMA(AsyncDma):

    includeList = ["pulp_core.h", "pulp_idma_utils.h"]

    _transferTemplates = {
        1:
            NodeTemplate("deeploy_idma_transfer_1d(${direction}, "
                         "(unsigned int)${ext}, (unsigned int)${loc}, ${length});"),
        2:
            NodeTemplate("deeploy_idma_transfer_2d(${direction}, (unsigned int)${ext}, "
                         "(unsigned int)${loc}, ${length}, ${strideExt}, ${strideLoc}, "
                         "${num_reps});"),
        3:
            NodeTemplate("deeploy_idma_transfer_3d(${direction}, (unsigned int)${ext}, "
                         "(unsigned int)${loc}, ${length}, ${strideExt}, ${strideLoc}, "
                         "${num_reps}, ${strideExt_3d}, ${strideLoc_3d}, ${num_reps_3d});")
    }
    _waitingStrategy = DirectionWaitingStrategy(iDMAChannelFuture, "channel_id")

    def __init__(
        self,
        transferTemplates: Optional[Dict[int, NodeTemplate]] = None,
    ) -> None:
        if transferTemplates is None:
            transferTemplates = dict(type(self)._transferTemplates)
        super().__init__(transferTemplates)

    def checkTransfer(
        self,
        ctxt: NetworkContext,
        externalBuffer: VariableBuffer,
        localBuffer: VariableBuffer,
        shape: Tuple[int, ...],
        strideExt: Tuple[int, ...],
        strideLoc: Tuple[int, ...],
        direction: DmaDirection,
    ) -> None:

        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        # 1D is split at runtime; in 2D/3D the row length is 16-bit (strides and reps are 32-bit)
        if len(shape) >= 2:
            length = shape[-1]
            assert length <= _MAX_LEN, (f"iDMA row length {length} exceeds the HAL's 16-bit limit ({_MAX_LEN})")
            assert strideExt[-2] >= length, "External stride must be at least equal to the length"
            assert strideLoc[-2] >= length, "Local stride must be at least equal to the length"

    def transferOpRepr(
        self,
        externalBuffer: VariableBuffer,
        localBuffer: VariableBuffer,
        shape: Tuple[int, ...],
        strideExt: Tuple[int, ...],
        strideLoc: Tuple[int, ...],
        direction: DmaDirection,
        future: Future,
    ) -> OperatorRepresentation:

        opRepr = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)

        opRepr["direction"] = _DIR_EXT_TO_LOC if direction == "ExternalToLocal" else _DIR_LOC_TO_EXT
        opRepr["length"] = shape[-1]
        # 2nd dimension (rows): stride between rows and number of rows
        if len(shape) >= 2:
            opRepr["strideExt"] = strideExt[-2]
            opRepr["strideLoc"] = strideLoc[-2]
            opRepr["num_reps"] = shape[-2]
        # 3rd dimension (planes): stride between planes and number of planes
        if len(shape) == 3:
            opRepr["strideExt_3d"] = strideExt[-3]
            opRepr["strideLoc_3d"] = strideLoc[-3]
            opRepr["num_reps_3d"] = shape[-3]

        return opRepr
