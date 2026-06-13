# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""DMA-safe PULP tile constraints for element-wise / transpose ops.

The generic ``UnaryTileConstraint`` and ``TransposeTileConstraint`` only relate
input and output dimensions; they place no lower bound on how small the tiled
*contiguous* extent may become. On PULP the L2<->L3 transfer is a 2D
``pi_cl_ram_copy_2d`` whose contiguous "line" length equals the tile's innermost
extent. When the tiler splits the innermost (contiguous) dimension and leaves a
1-element remainder (e.g. a 101-wide dim split 100 + 1), that line collapses to a
single element and ``pi_cl_ram_copy_wait`` never completes -> the network hangs
(reproduced in GVSoC; see memory `qtsdr-l3-transpose-hang`).

The fix uses the tiler's ``addMinTileSizeConstraint`` to require that, *if* the
innermost non-unit dimension is tiled, the remainder tile is at least
``_MIN_INNERMOST_TILE`` elements wide. Unlike pinning the dim to full size (which
over-constrains and can make tiling infeasible), this still lets the tiler split
the contiguous dim -- it just never produces a degenerate single-element line.
"""

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.TileConstraints.TransposeTileConstraint import TransposeTileConstraint
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryTileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel

# Minimum width (in elements) of any tile along the innermost contiguous dim, so
# the resulting 2D DMA line is never a single element.
_MIN_INNERMOST_TILE = 2


def _innermostNonUnitDim(shape) -> int:
    """Index of the innermost (last) dimension with size > 1, or None if all unit."""
    for dim in range(len(shape) - 1, -1, -1):
        if shape[dim] > 1:
            return dim
    return None


def _constrainInnermostDim(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext, bufferName: str) -> None:
    """Forbid a degenerate (sub-`_MIN_INNERMOST_TILE`) tile on the innermost dim."""
    shape = ctxt.lookup(bufferName).shape
    dim = _innermostNonUnitDim(shape)
    if dim is None or shape[dim] <= _MIN_INNERMOST_TILE:
        return
    dimVar = tilerModel.getTensorDimVar(tensorName = bufferName, dimIdx = dim)
    # addMinTileSizeConstraint needs the full dim size under a parseDict key; add a
    # uniquely-named one (unused by codegen) so we don't disturb existing keys.
    safeName = bufferName.replace(".", "_")
    sizeKey = f"_dmaSafeDim_{safeName}_{dim}"
    parseDict[sizeKey] = int(shape[dim])
    tilerModel.addMinTileSizeConstraint(parseDict, sizeKey, dimVar, _MIN_INNERMOST_TILE,
                                        prefix = f"dmasafe_{safeName}_{dim}_")


class PULPUnaryTileConstraint(UnaryTileConstraint):
    """UnaryTileConstraint that forbids a degenerate innermost-dim tile."""

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = UnaryTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)
        for bufferName in [parseDict['data_in'], parseDict['data_out']]:
            _constrainInnermostDim(tilerModel, parseDict, ctxt, bufferName)
        return tilerModel


class PULPTransposeTileConstraint(TransposeTileConstraint):
    """TransposeTileConstraint that forbids a degenerate innermost-dim tile."""

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = TransposeTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)
        for bufferName in [parseDict['data_in'], parseDict['data_out']]:
            _constrainInnermostDim(tilerModel, parseDict, ctxt, bufferName)
        return tilerModel
