# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union
from ortools.constraint_solver.pywrapcp import IntVar

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateFlatOffset, stridesFromShape


class EggrollTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['shape_in']
        inputBuffer = ctxt.lookup(inputBufferName)
        outputBufferName = parseDict['data_out']
        outputBuffer = ctxt.lookup(outputBufferName)
        inputDimVar0 = int(inputBuffer.values[0])
        inputDimVar1 = int(inputBuffer.values[1])
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # for dim in range(inputBuffer.values[0]):
        #     inputDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = dim)
        for dim in range(len(outputBuffer.shape)):
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
            if dim == 0:
                tilerModel.addConstraint(outputDimVar <= inputDimVar0)
            elif dim == 1:
                tilerModel.addConstraint(outputDimVar <= inputDimVar1)
        return tilerModel
    
    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBufferName = parseDict['data_out']
        inputBuffer = ctxt.lookup(inputBufferName)

        Dim0Idx = 0
        Dim1Idx = 1
        symbolicParseDict = parseDict.copy()
        symbolicParseDict['inputDimVar0'] = tilerModel.getTensorDimVar(inputBuffer.name, Dim0Idx)
        symbolicParseDict['inputDimVar1'] = tilerModel.getTensorDimVar(inputBuffer.name, Dim1Idx)
        return symbolicParseDict
    
    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['shape_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        
        replacements = {"inputDimVar0": [], "inputDimVar1": [], "size": [], "tile_seed_offset": []}
        replacementTypes = {"inputDimVar0": PointerClass(uint32_t), "inputDimVar1": PointerClass(uint32_t), "size": PointerClass(uint32_t), "tile_seed_offset": PointerClass(uint32_t)}

        # Per-tile global element offset, so the RNG seed differs between tiles.
        outShape = ctxt.lookup(operatorRepresentation['data_out']).shape
        if isinstance(outShape, int):  # 1-D buffers store shape as a bare int
            outShape = (outShape,)
        outStrides = stridesFromShape(outShape)

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)
            replacements['inputDimVar0'].append(cube.dims[0])
            replacements['inputDimVar1'].append(cube.dims[1])

        for absCube in absoluteOutputCubes:
            replacements['tile_seed_offset'].append(calculateFlatOffset(absCube.absoluteOffset, outStrides))


        inputLoadSchedule = []
        outputLoadSchedule = []

        # for cube in outputCubes:
        #     inputLoadSchedule.append({"shape_in": cube})
            
        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
