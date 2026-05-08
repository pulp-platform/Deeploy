# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


class SGDTileConstraint(BOPTileConstraint):

    dataIn1Name = 'weight'
    dataIn2Name = 'grad'
    dataOutName = 'weight_updated'

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = super().addGeometricalConstraint(tilerModel, parseDict, ctxt)

        # Force spatial dims (index >= 2) to full size so that minimizeRectangle
        # can collapse them and DMA tiles stay rank ≤ 2 (L3Dma pi_cl_ram_copy_2d limit).
        weightName = parseDict[cls.dataIn1Name]
        shape = ctxt.lookup(weightName).shape
        if not isinstance(shape, int) and len(shape) > 2:
            for dimIdx in range(2, len(shape)):
                dimVar = tilerModel.getTensorDimVar(tensorName = weightName, dimIdx = dimIdx)
                tilerModel.addConstraint(dimVar == shape[dimIdx])

        return tilerModel


class ReluGradTileConstraint(BOPTileConstraint):

    dataIn1Name = 'grad_out'
    dataIn2Name = 'data_in'
    dataOutName = 'grad_in'
