# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


_TILED_TENSORS = ['X', 'G', 'V', 'V_new']
_ARRAY_INPUT_TENSORS = ['X', 'G', 'V']


class AdamUpdateVTileConstraint(TileConstraint):

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> TilerModel:

        for name in _TILED_TENSORS:
            tilerModel.addTensorDimToModel(ctxt, parseDict[name])

        xShape = ctxt.lookup(parseDict['X']).shape

        for dim in range(len(xShape)):
            xDimVar = tilerModel.getTensorDimVar(tensorName = parseDict['X'], dimIdx = dim)
            for name in ['G', 'V', 'V_new']:
                dimVar = tilerModel.getTensorDimVar(tensorName = parseDict[name], dimIdx = dim)
                tilerModel.addConstraint(xDimVar == dimVar)

        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict,
                            ctxt: NetworkContext) -> TilerModel:
        xShape = ctxt.lookup(parseDict['X']).shape
        for dim in range(1, len(xShape)):
            dimVar = tilerModel.getTensorDimVar(tensorName = parseDict['X'], dimIdx = dim)
            tilerModel.addConstraint(dimVar == xShape[dim])
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint,
            absoluteOutputCubes: List[AbsoluteHyperRectangle], targetMemLevel: str,
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = _TILED_TENSORS
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}
        replacementTypes = {"size": PointerClass(uint16_t)}

        for cubeAbs in absoluteOutputCubes:
            cube = cubeAbs.rectangle
            replacements["size"].append(int(np.prod(cube.dims)))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            tile_load = {name: cube for name in _ARRAY_INPUT_TENSORS}
            inputLoadSchedule.append(tile_load)

        for cube in outputCubes:
            outputLoadSchedule.append({'V_new': cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule,
                                        outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
