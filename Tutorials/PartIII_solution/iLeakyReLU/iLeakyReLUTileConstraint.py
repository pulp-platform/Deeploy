# ----------------------------------------------------------------------
# File: iLeakyReLUTileConstraint.py
#
# SoCDAML Part III - TA reference solution.
# Tiling + performance constraint for the iLeakyReLU op.
#
# Drop this file into:
#   Deeploy/Targets/PULPOpen/TileConstraints/iLeakyReLUTileConstraint.py
# ----------------------------------------------------------------------
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryTileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


class iLeakyReLUTileConstraint(UnaryTileConstraint):
    """
    Geometry is inherited from UnaryTileConstraint (input shape == output
    shape per axis; one shared cube per output tile). On top of that we
    add the Step 6a performance constraint: the innermost (last) tile
    dim must be a multiple of 16 so the 4-byte SIMD kernel can vectorize
    the per-core chunk without a tail iteration.
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = UnaryTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)

        inputBufferName = parseDict['data_in']
        inputShape = ctxt.lookup(inputBufferName).shape
        lastDim = len(inputShape) - 1
        lastDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDim)

        # Force the tiled inner dimension to be a multiple of 16. This
        # ensures (per-core chunk) is a multiple of 4 once split across
        # 8 cores -> the v4s SIMD inner loop is always tail-free.
        #
        # NOTE: this must be addTileSizeDivisibleConstraint, not
        # addMinTileSizeConstraint. The latter only forces the *leftover*
        # tile to be at least `modulo` elements
        # Both read parseDict[varName] as the original axis size, so
        # inject it here.
        if inputShape[lastDim] >= 16:
            dimKey = f'dim_{lastDim}'
            parseDict[dimKey] = int(inputShape[lastDim])
            tilerModel.addTileSizeDivisibleConstraint(parseDict, dimKey, lastDimVar, 16)

        return tilerModel
