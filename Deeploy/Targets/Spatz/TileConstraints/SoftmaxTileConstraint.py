# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, TransientBuffer
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class SoftmaxTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Register and pin all referenced tensors to full shape to avoid tiling.
        # This also covers constant inputs that may appear as parseDict string references.
        tensorNames: List[str] = []

        for value in parseDict.values():
            if not isinstance(value, str):
                continue
            if ctxt.is_global(value) or ctxt.is_local(value):
                tensorNames.append(value)

        for tensorName in tensorNames:
            _buffer = ctxt.lookup(tensorName)
            if isinstance(_buffer, TransientBuffer):
                continue

            tilerModel.addTensorDimToModel(ctxt, tensorName)

            for idx, shapeDim in enumerate(_buffer.shape):
                tileDimVar = tilerModel.getTensorDimVar(tensorName = tensorName, dimIdx = idx)
                tilerModel.addConstraint(tileDimVar == shapeDim)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        symbolicParseDict = parseDict.copy()

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        dataInBuffer = ctxt.lookup(operatorRepresentation['data_in'])

        dataInCube = HyperRectangle(offset = (0,) * len(dataInBuffer.shape), dims = tuple(dataInBuffer.shape))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for out in outputCubes:
            inputLoadSchedule.append({'data_in': dataInCube})
            outputLoadSchedule.append({'data_out': out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        repScheme = VariableReplacementScheme({}, {})

        return repScheme, schedule
