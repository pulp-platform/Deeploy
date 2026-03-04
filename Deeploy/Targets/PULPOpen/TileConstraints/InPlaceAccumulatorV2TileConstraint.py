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

        # The tiled template (tiledReferenceTemplate) writes ONLY to ${accum_buffer}
        # and does NOT reference ${data_out}.  Therefore data_out is omitted from
        # addrNames — it gets no L1 tile ref and generates no DMA transfer.
        #
        # Background: the memory allocator may place data_out at a DIFFERENT L2 address
        # from accum_buffer, even though they are declared as aliases.  If data_out were
        # added to outputBaseOffsets + outputLoadSchedule, the egress DMA would write the
        # full weight tensor (with a stride) starting at data_out's L2 address, corrupting
        # other live L2 buffers that share that region.
        #
        # The optimizer reads the updated gradient from accum_buffer's L2 address
        # (DeeployNetwork_inputs[TRAINING_GRAD_BUF_START_IDX + wi]), which is correctly
        # updated by the accum_buffer egress DMA below.
        addrNames = [cls.dataIn1Name, cls.dataIn2Name, 'lazy_reset_grad']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # Add accum_buffer to outputBaseOffsets + outputLoadSchedule for the in-place
        # write-back egress DMA (L1 tile → accum_buffer's L2 address).
        outputBaseOffsets[cls.dataIn1Name] = inputBaseOffsets[cls.dataIn1Name]

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
            # Egress: write accum_buffer tile back to its L2 address (input_4 / input_5).
            outputLoadSchedule.append({
                cls.dataIn1Name: out,
            })

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
