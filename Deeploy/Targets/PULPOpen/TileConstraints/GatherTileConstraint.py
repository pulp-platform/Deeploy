# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, TransientBuffer
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class GatherTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        pointer: List[str] = []

        for key, value in parseDict.items():
            if not isinstance(value, str):
                continue

            if ctxt.is_global(value) or ctxt.is_local(value):
                pointer.append(value)

        for tensorName in pointer:

            _buffer = ctxt.lookup(tensorName)
            if isinstance(_buffer, TransientBuffer):
                continue

            tilerModel.addTensorDimToModel(ctxt, tensorName)

            for idx, shapeDim in enumerate(_buffer.shape):
                tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = tensorName, dimIdx = idx) == shapeDim)

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

        schedule = TilingSchedule({}, {}, [], [])
        repScheme = VariableReplacementScheme({}, {})

        return repScheme, schedule
