# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class InPlaceAccumulatorV2TileConstraint(BOPTileConstraint):
    """Tile constraint for InPlaceAccumulatorV2.

    Tiles buffer and gradient together (same shape); lazy_reset_grad is a
    scalar (1 element) and is not tiled.
    """

    dataIn1Name = 'accum_buffer'
    dataIn2Name = 'gradient'
    dataOutName = 'data_out'

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Register buffer, gradient, data_out and add BOP equality constraints
        tilerModel = super().addGeometricalConstraint(tilerModel, parseDict, ctxt)

        # Register lazy_reset_grad (scalar flag, not tiled): fix all dims to full size
        lazyResetName = parseDict['lazy_reset_grad']
        tilerModel.addTensorDimToModel(ctxt, lazyResetName)
        lazyResetTensor = ctxt.lookup(lazyResetName)
        shape = lazyResetTensor.shape
        dims = [shape] if isinstance(shape, int) else shape
        for idx, dim in enumerate(dims):
            dimVar = tilerModel.getTensorDimVar(lazyResetName, idx)
            tilerModel.addConstraint(dimVar == dim)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Egress strategy: use data_out (the proper graph output, present in
        # outputTensorMemoryConstraints) rather than accum_buffer (a graph input,
        # only in inputTensorMemoryConstraints).  This avoids two core-class issues:
        #   1. accum_buffer appearing in BOTH inputBaseOffsets and outputBaseOffsets
        #      causes a duplicate-hoist KeyError in TilingVariableReplacement.
        #   2. The egress DMA lookup uses outputTensorMemoryConstraints; accum_buffer
        #      is not there and would raise a KeyError.
        #
        # The trick: force outputBaseOffsets[data_out] to the SAME L1 arena offset as
        # inputBaseOffsets[accum_buffer].  Both data_out_ref and accum_buffer_ref then
        # map to the same physical L1 address.  The tiled kernel writes to ${accum_buffer}
        # (= accum_buffer_ref in L1); the egress DMA transfers data_out_ref (same L1
        # bytes) to data_out's L2 address, which is what the optimizer reads.
        addrNames = [cls.dataIn1Name, cls.dataIn2Name, cls.dataOutName, 'lazy_reset_grad']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # Pin data_out's L1 tile to the same arena slot as accum_buffer's L1 tile.
        outputBaseOffsets[cls.dataOutName] = inputBaseOffsets[cls.dataIn1Name]

        replacements = {"size": []}
        replacementTypes = {"size": PointerClass(uint16_t)}

        lazyResetName = operatorRepresentation['lazy_reset_grad']
        lazyResetShape = ctxt.lookup(lazyResetName).shape
        lazyResetDims = (lazyResetShape,) if isinstance(lazyResetShape, int) else tuple(lazyResetShape)
        lazyResetCube = HyperRectangle((0,) * len(lazyResetDims), lazyResetDims)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            replacements["size"].append(int(np.prod(cube.dims)))
            inputLoadSchedule.append({
                cls.dataIn1Name: cube,
                cls.dataIn2Name: cube,
                'lazy_reset_grad': lazyResetCube,
            })

        for out in outputCubes:
            # Egress: DMA from data_out_ref (same L1 slot as accum_buffer_ref) → data_out L2.
            outputLoadSchedule.append({
                cls.dataOutName: out,
            })

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
