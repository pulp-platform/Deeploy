# ----------------------------------------------------------------------
#
# File: MatMulTileConstraint.py
#
# Last edited: 28.04.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author: [Your Name]
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, uint16_t, uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel, PerformanceHint
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class RedmuleMatmulTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        # Add I/O dimensions to the model as variables
        for _buffer in [bufferA, bufferB, outputBuffer]:
            tilerModel.addTensorDimToModel(ctxt, _buffer.name)

        tensorsShapeLen = len(bufferA.shape)

        AFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                  dimIdx = (tensorsShapeLen - 2) + parseDict['transA'])
        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                  dimIdx = (tensorsShapeLen - 2) + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transB'])
        outputFirstDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = (tensorsShapeLen - 2))
        outputSecondDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = (tensorsShapeLen - 1))

        # Map output dims to inputs dims
        for idx in range(tensorsShapeLen - 2):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = bufferA.name, dimIdx = idx))
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = bufferB.name, dimIdx = idx))

        tilerModel.addConstraint(outputFirstDimVar == AFirstDimVar)
        tilerModel.addConstraint(outputSecondDimVar == BSecondDimVar)

        # Add GEMM Geometrical constraints
        tilerModel.addConstraint(ASecondDimVar == BFirstDimVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])

        tensorsShapeLen = len(bufferA.shape)

        AFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                  dimIdx = (tensorsShapeLen - 2) + parseDict['transA'])
        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                  dimIdx = (tensorsShapeLen - 2) + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transB'])

        # VIC: We don't want to deal with intermediate results between kernel calls
        tilerModel.addConstraint(ASecondDimVar == parseDict['N'])
        tilerModel.addConstraint(BFirstDimVar == parseDict['N'])


        # Hardware-specific constraints for 4x12 accelerator
        tilerModel.addConstraint(BSecondDimVar == BSecondDimVar.Max(), strategy = PerformanceHint(1))

        M_full_size = ctxt.lookup(bufferA.name).shape[(tensorsShapeLen - 2) + parseDict['transA']]
        if M_full_size >= 4:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "M",
                                                      AFirstDimVar,
                                                      4,
                                                      strategy = PerformanceHint(priority = 1))
        else:
            tilerModel.addConstraint(AFirstDimVar == AFirstDimVar.Max(), strategy = PerformanceHint(1))

        N_full_size = ctxt.lookup(bufferB.name).shape[(tensorsShapeLen - 2) + parseDict['transB']]
        if N_full_size >= 12:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "O",
                                                      BSecondDimVar,
                                                      12,
                                                      strategy = PerformanceHint(priority = 1))
        else:
            tilerModel.addConstraint(BSecondDimVar == BSecondDimVar.Max(), strategy = PerformanceHint(1))
        
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['A', 'B', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varA = operatorRepresentation['A']

        NSize = ctxt.lookup(varA).shape[-1]
        NOffset = 0

        inputACubes = []
        inputBCubes = []

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
            BCube = HyperRectangle((BatchOffset, BOffset, NOffset, OOffset), (BatchSize, BSize, NSize, OSize))

            inputACubes.append(ACube)
            inputBCubes.append(BCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        replacements["N"] = [NSize] * len(outputCubes)

        replacementTypes = {
            "M": PointerClass(int8_t),
            "N": PointerClass(int8_t),
            "O": PointerClass(int8_t),
            "batch": PointerClass(int8_t)
        }

        for a, b in zip(inputACubes, inputBCubes):
            inputLoadSchedule.append({"A": a, "B": b})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule
