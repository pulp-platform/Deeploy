# ----------------------------------------------------------------------
# File: iLeakyReLUTileConstraint.py  (SoCDAML Part III - Step 5+7a skeleton)
#
# Drop this file into:
#   Deeploy/Targets/PULPOpen/TileConstraints/iLeakyReLUTileConstraint.py
#
# UnaryTileConstraint already implements the geometry and serializer
# you need for an elementwise op. You only have to subclass it. In
# Step 6a you'll add a performance constraint on top.
# ----------------------------------------------------------------------
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryTileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


class iLeakyReLUTileConstraint(UnaryTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = UnaryTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)

        # TODO(student, Step 6a): add a performance constraint so the
        # innermost tile dim is a multiple of 16. Helpful API:
        #     tilerModel.addMinTileSizeConstraint(parseDict, name,
        #                                         tensorDimVar, modulo)
        # See: Deeploy/Targets/Generic/TileConstraints/ConvTileConstraint.py
        # for a usage example.

        return tilerModel
