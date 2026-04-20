# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 Add tile constraint — extends generic BOPTileConstraint with divisibility."""

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint
from Deeploy.Targets.XDNA2.TileConstraints.DivisibilityHelper import addDivisibilityConstraints
from Deeploy.TilingExtension.TilerModel import TilerModel


class XDNA2AddTileConstraint(BOPTileConstraint):

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = super().addGeometricalConstraint(tilerModel, parseDict, ctxt)
        addDivisibilityConstraints(tilerModel, parseDict[cls.dataOutName], ctxt)
        return tilerModel
