# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class SliceTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get necessary information
        #   Get I/O buffer names
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        #   Get I/O shapes
        inputShape = parseDict['data_in_shape']

        #   Get other necessary information
        sliceStarts = parseDict['starts']
        sliceEnds = parseDict['ends']
        sliceAxes = parseDict['axes']
        sliceSteps = parseDict['steps']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # Add constratints for the I/O dimensions
        for idx in range(len(inputShape)):
            inputDimensionVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx)
            outputDimensionVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx)

            if idx in sliceAxes:
                axIndex = list(sliceAxes).index(idx)
                axStart = sliceStarts[axIndex]
                axEnd = sliceEnds[axIndex]
                axStep = sliceSteps[axIndex]

                # FIXME: COULD BE AN ISSUE
                # tilerModel.addConstraint(axStep * (outputDimensionVar - 1) < inputDimensionVar)
                # tilerModel.addConstraint(inputDimensionVar <= axStep * outputDimensionVar)
                # tilerModel.addConstraint(inputDimensionVar <= abs(axEnd - axStart))

            else:
                tilerModel.addConstraint(outputDimensionVar == inputDimensionVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # TODO
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        # TODO
        symbolicParseDict = parseDict.copy()

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        # TODO

        # Prepare address names
        addrNames = ['data_in', 'data_out']

        # Extract memory base addresses for each of the required components,
        # based on the computed memory configuration
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # Prepare replacement lists for the elements inside the operator representation,
        # for the cubes to be computed further down in this function
        replacements: Dict[str, List[int]] = {}

        replacementTypes = {}

        # Prepare loading schedule lists
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Prepare containing objects with information computed in this function regarding tiling schedule
        # and variable replacement inside operator representation
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
