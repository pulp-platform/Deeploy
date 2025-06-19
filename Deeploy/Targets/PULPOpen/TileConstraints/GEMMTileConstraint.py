# ----------------------------------------------------------------------
#
# File: GEMMTileConstraint.py
#
# Last edited: 02.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
# - Moritz Scherer, scheremo@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class GEMMTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])
        bufferC = ctxt.lookup(name = parseDict['C'])  # Add from RequantShift
        mulBuffer = ctxt.lookup(name = parseDict['mul'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        # Add I/O dimensions to the model as variables
        for bufferName in [bufferA.name, bufferB.name, bufferC.name, mulBuffer.name, outputBuffer.name]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        dimOffsetA = len(bufferA.shape) - 2
        dimOffsetB = len(bufferB.shape) - 2
        dimOffsetOut = len(outputBuffer.shape) - 2

        AFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = dimOffsetA + parseDict['transA'])
        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = dimOffsetA + 1 - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name, dimIdx = dimOffsetB + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = dimOffsetB + 1 - parseDict['transB'])
        outputFirstDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = dimOffsetOut)
        outputSecondDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = dimOffsetOut + 1)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputFirstDimVar == AFirstDimVar)
        tilerModel.addConstraint(outputSecondDimVar == BSecondDimVar)

        # Add GEMM Geometrical constraints
        tilerModel.addConstraint(ASecondDimVar == BFirstDimVar)

        mulDimVar = tilerModel.getTensorDimVar(tensorName = mulBuffer.name, dimIdx = 0)
        addDimVar = tilerModel.getTensorDimVar(tensorName = bufferC.name, dimIdx = 0)

        tilerModel.addConstraint(outputSecondDimVar == mulDimVar)
        tilerModel.addConstraint(outputSecondDimVar == addDimVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])

        dimOffsetA = len(bufferA.shape) - 2
        dimOffsetB = len(bufferB.shape) - 2

        AFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = dimOffsetA + parseDict['transA'])

        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = dimOffsetA + 1 - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name, dimIdx = dimOffsetB + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = dimOffsetB + 1 - parseDict['transB'])

        # VIC: We don't want to deal with intermediate results between kernel calls
        tilerModel.addConstraint(ASecondDimVar == parseDict['N'])
        tilerModel.addConstraint(BFirstDimVar == parseDict['N'])

        if (parseDict["O"] >= 16):
            #modulus = tilerModel.addMinTileSizeConstraint(parseDict, 'O', BSecondDimVar, 8, prefix = "8_")
            modulus = tilerModel.addTileSizeDivisibleConstraint(parseDict, 'O', BSecondDimVar, 16, prefix = "16_")

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['A', 'B', 'mul', 'C', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)
        varA = operatorRepresentation['A']

        NSize = ctxt.lookup(varA).shape[-1]
        NOffset = 0

        inputACubes = []
        inputBCubes = []
        inputMulCubes = []
        inputAddCubes = []

        replacements = {"M": [], "O": [], "batch": []}

        # Every output is constructed by a pair of inputs. Reconstruct this pair.
        for cube in outputCubes:

            BSize = 1
            BOffset = 0
            BatchSize = 1
            BatchOffset = 0

            if len(cube.offset) == 2:
                (MOffset, OOffset) = cube.offset
                (MSize, OSize) = cube.dims
            elif len(cube.offset) == 3:
                (BatchOffset, MOffset, OOffset) = cube.offset
                (BatchSize, MSize, OSize) = cube.dims
            else:
                (BatchOffset, BOffset, MOffset, OOffset) = cube.offset
                (BatchSize, BSize, MSize, OSize) = cube.dims

            replacements["M"].append(MSize)
            replacements["O"].append(OSize)
            replacements["batch"].append(BSize)

            ACube = HyperRectangle((BatchOffset, BOffset, MOffset, NOffset), (BatchSize, BSize, MSize, NSize))
            BCube = HyperRectangle((BatchOffset, BOffset, OOffset, NOffset), (BatchSize, BSize, OSize, NSize))

            RequantCube = HyperRectangle((OOffset,), (OSize,))

            inputACubes.append(ACube)
            inputBCubes.append(BCube)
            inputMulCubes.append(RequantCube)
            inputAddCubes.append(RequantCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        replacements["N"] = [NSize] * len(outputCubes)

        replacementTypes = {
            "M": PointerClass(uint16_t),
            "N": PointerClass(uint16_t),
            "O": PointerClass(uint16_t),
            "batch": PointerClass(uint8_t)
        }

        for a, b, c, mul in zip(inputACubes, inputBCubes, inputAddCubes, inputMulCubes):
            inputLoadSchedule.append({"A": a, "B": b, "C": c, "mul": mul})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule


class MatrixVecTileConstraint(GEMMTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        tm = GEMMTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)

        return tm

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        tm = GEMMTileConstraint.addPolicyConstraint(tilerModel, parseDict, ctxt)

        return tm


class TallGEMMTileConstraint(GEMMTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        tm = GEMMTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)

        return tm

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        tm = GEMMTileConstraint.addPolicyConstraint(tilerModel, parseDict, ctxt)

        return tm


class FloatGEMMTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])
        
        has_bias = 'C' in parseDict and parseDict['C'] is not None
        bufferC = None
        if has_bias:
            bufferC = ctxt.lookup(name = parseDict['C'])

        # Add I/O dimensions to the model as variables
        buffer_names = [bufferA.name, bufferB.name, outputBuffer.name]
        if has_bias:
            buffer_names.append(bufferC.name)
            
        for bufferName in buffer_names:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        dimOffsetA = len(bufferA.shape) - 2
        dimOffsetB = len(bufferB.shape) - 2
        dimOffsetOut = len(outputBuffer.shape) - 2

        AFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = dimOffsetA + parseDict['transA'])
        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = dimOffsetA + 1 - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name, dimIdx = dimOffsetB + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = dimOffsetB + 1 - parseDict['transB'])
        outputFirstDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = dimOffsetOut)
        outputSecondDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = dimOffsetOut + 1)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputFirstDimVar == AFirstDimVar)
        tilerModel.addConstraint(outputSecondDimVar == BSecondDimVar)

        # Add GEMM Geometrical constraints
        tilerModel.addConstraint(ASecondDimVar == BFirstDimVar)

        # Add bias constraints only if bias is present
        if has_bias:
            dimOffsetC = len(bufferC.shape) - 2
            addDimVar_1 = tilerModel.getTensorDimVar(tensorName = bufferC.name, dimIdx = dimOffsetC)
            addDimVar_2 = tilerModel.getTensorDimVar(tensorName = bufferC.name, dimIdx = dimOffsetC + 1)
            tilerModel.addConstraint(outputFirstDimVar == addDimVar_1)
            tilerModel.addConstraint(outputSecondDimVar == addDimVar_2)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])

        dimOffsetA = len(bufferA.shape) - 2
        dimOffsetB = len(bufferB.shape) - 2

        AFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = dimOffsetA + parseDict['transA'])

        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = dimOffsetA + 1 - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name, dimIdx = dimOffsetB + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = dimOffsetB + 1 - parseDict['transB'])

        # VIC: We don't want to deal with intermediate results between kernel calls
        tilerModel.addConstraint(ASecondDimVar == parseDict['N'])
        tilerModel.addConstraint(BFirstDimVar == parseDict['N'])

        if (parseDict["O"] >= 16):
            # modulus = tilerModel.addMinTileSizeConstraint(parseDict, 'O', BSecondDimVar, 8, prefix="8_")
            modulus = tilerModel.addTileSizeDivisibleConstraint(parseDict, 'O', BSecondDimVar, 16, prefix = "16_")

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Check if C (bias) is present
        has_bias = 'C' in operatorRepresentation and operatorRepresentation['C'] is not None
        
        # Build address names list based on whether bias is present
        addrNames = ['A', 'B', 'data_out']
        if has_bias:
            addrNames.insert(2, 'C')  # Insert 'C' before 'data_out'

        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        transA = operatorRepresentation['transA']
        transB = operatorRepresentation['transB']

        varA = operatorRepresentation['A']
        varB = operatorRepresentation['B']

        if transA == 0:
            NSize = ctxt.lookup(varA).shape[-1]
        else:
            NSize = ctxt.lookup(varA).shape[-2]

        NOffset = 0

        inputACubes = []
        inputBCubes = []
        inputAddCubes = []

        replacements = {"M": [], "O": [], "batch": []}

        # Every output is constructed by a pair of inputs. Reconstruct this pair.
        for cube in outputCubes:

            BSize = 1
            BOffset = 0
            BatchSize = 1
            BatchOffset = 0

            if len(cube.offset) == 2:
                (MOffset, OOffset) = cube.offset
                (MSize, OSize) = cube.dims
            elif len(cube.offset) == 3:
                (BatchOffset, MOffset, OOffset) = cube.offset
                (BatchSize, MSize, OSize) = cube.dims
            else:
                (BatchOffset, BOffset, MOffset, OOffset) = cube.offset
                (BatchSize, BSize, MSize, OSize) = cube.dims

            replacements["M"].append(MSize)
            replacements["O"].append(OSize)
            replacements["batch"].append(BSize)

            if transA == 0:
                ACube = HyperRectangle((BatchOffset, BOffset, MOffset, NOffset), (BatchSize, BSize, MSize, NSize))
            else:
                ACube = HyperRectangle((BatchOffset, BOffset, NOffset, MOffset), (BatchSize, BSize, NSize, MSize))

            if transB == 0:
                BCube = HyperRectangle((BatchOffset, BOffset, NOffset, OOffset), (BatchSize, BSize, NSize, OSize))
            else:
                BCube = HyperRectangle((BatchOffset, BOffset, OOffset, NOffset), (BatchSize, BSize, OSize, NSize))

            inputACubes.append(ACube)
            inputBCubes.append(BCube)
            
            # Only create C cubes if bias is present
            if has_bias:
                CCube = HyperRectangle(cube.offset, cube.dims)
                inputAddCubes.append(CCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        replacements["N"] = [NSize] * len(outputCubes)

        replacementTypes = {
            "M": PointerClass(uint16_t),
            "N": PointerClass(uint16_t),
            "O": PointerClass(uint16_t),
            "batch": PointerClass(uint8_t)
        }

        # Build input load schedule based on whether bias is present
        if has_bias:
            for a, b, c in zip(inputACubes, inputBCubes, inputAddCubes):
                inputLoadSchedule.append({"A": a, "B": b, "C": c})
        else:
            for a, b in zip(inputACubes, inputBCubes):
                inputLoadSchedule.append({"A": a, "B": b})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule
