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

class GatherTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        pointer: List[str] = []
        for key, value in parseDict.items():
            if not isinstance(value, str):
                continue

            if ctxt.is_global(value) or ctxt.is_local(value):
                pointer.append(value)

                _buffer = ctxt.lookup(value)
                if isinstance(_buffer, TransientBuffer):
                    continue

                tilerModel.addTensorDimToModel(ctxt, value)

                # no tile contraint for data_in, because is not moved by the tiling engine
                if key == 'data_in':
                    continue

                for idx, shapeDim in enumerate(_buffer.shape):
                    tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = value, dimIdx = idx) == shapeDim)

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

        # Dynamic-DMA Gather policy:
        # - DMA only indices into local memory
        # - Do NOT DMA the full data_in tile into local memory
        # - DMA the output tile back to external memory
        addrNames = ['indices', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        dataInBuffer = ctxt.lookup(operatorRepresentation['data_in'])
        indicesBuffer = ctxt.lookup(operatorRepresentation['indices'])

        dataInCube = HyperRectangle(offset = (0,) * len(dataInBuffer.shape), dims = tuple(dataInBuffer.shape))
        indicesCube = HyperRectangle(offset = (0,) * len(indicesBuffer.shape), dims = tuple(indicesBuffer.shape))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for out in outputCubes:
            # Gather execution policy (dynamic DMA): load indices in L1, execute once, then store output tile.
            # data_in stays in external memory; selected rows are fetched directly into the local output buffer.
            _ = dataInCube  # Keep for clarity; intentionally unused in this schedule.
            inputLoadSchedule.append({'indices': indicesCube})
            outputLoadSchedule.append({'data_out': out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        repScheme = VariableReplacementScheme({}, {})

        return repScheme, schedule
