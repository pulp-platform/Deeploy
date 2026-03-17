# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, TransientBuffer
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class UntiledTileConstraint(TileConstraint):

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

            shape = [_buffer.shape] if isinstance(_buffer.shape, int) else _buffer.shape
            for idx, shapeDim in enumerate(shape):
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
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        schedule = TilingSchedule({}, {}, [], [])
        repScheme = VariableReplacementScheme({}, {})

        for key, value in tilingSolution.tensorMemoryConstraints.items():
            _buffer = ctxt.lookup(key)
            if isinstance(_buffer, TransientBuffer):
                continue

            fullShape = tuple(_buffer.shape)
            memoryConstraints = list(value.memoryConstraints.values())

            # Untiled tensors may still be materialized on multiple memory levels when the same
            # full-shape buffer is replicated along the memory path (for example, an untiled Slice
            # fed by an L3-backed producer). This is not true tiling and should remain acceptable
            # here. We only reject the case where different memory levels carry different shapes.
            assert all(memValue.shape == fullShape for memValue in memoryConstraints), \
                f"{cls} should be untiled, but {value} carries multiple shapes across memory levels!"

        return repScheme, schedule
