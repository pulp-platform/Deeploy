# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 Unary tile constraint — extends generic UnaryTileConstraint with divisibility."""

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryTileConstraint
from Deeploy.Targets.XDNA2.TileConstraints.DivisibilityHelper import addDivisibilityConstraints
from Deeploy.TilingExtension.TilerModel import TilerModel


class XDNA2UnaryTileConstraint(UnaryTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = UnaryTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)
        addDivisibilityConstraints(tilerModel, parseDict['data_out'], ctxt)
        return tilerModel
