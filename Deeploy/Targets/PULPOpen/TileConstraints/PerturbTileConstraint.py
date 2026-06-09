# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateFlatOffset, stridesFromShape


class PerturbTileConstraint(UnaryTileConstraint):
    """Tile constraint for the floating-point ZO perturbation kernels.

    Identical geometry to UnaryTileConstraint, but additionally exposes a per-tile
    ``tile_seed_offset`` (the flat global element offset of each tile's first element)
    so the kernel seed differs between tiles. Without it, the identical kernel code
    emitted for every tile would reuse the same RNG seed, making the perturbation
    periodic with period equal to the tile size instead of i.i.d. across the weight.
    """

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        variableReplacementSchedule, tilingSchedule = super().serializeTilingSolution(
            tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation)

        outShape = ctxt.lookup(operatorRepresentation['data_out']).shape
        if isinstance(outShape, int):  # 1-D buffers store shape as a bare int
            outShape = (outShape,)
        outStrides = stridesFromShape(outShape)
        variableReplacementSchedule.perTileReplacements['tile_seed_offset'] = [
            calculateFlatOffset(absCube.absoluteOffset, outStrides) for absCube in absoluteOutputCubes
        ]
        variableReplacementSchedule.replacementTypes['tile_seed_offset'] = PointerClass(uint32_t)

        return variableReplacementSchedule, tilingSchedule
