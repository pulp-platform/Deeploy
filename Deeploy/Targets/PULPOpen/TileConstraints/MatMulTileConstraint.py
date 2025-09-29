# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class MatMulTileConstraint(TileConstraint):

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
                                                  dimIdx = (tensorsShapeLen - 2) + int(parseDict['transA']))
        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = (tensorsShapeLen - 1) - int(parseDict['transA']))
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                  dimIdx = (tensorsShapeLen - 2) + int(parseDict['transB']))
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = (tensorsShapeLen - 1) - int(parseDict['transB']))
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

        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                  dimIdx = (tensorsShapeLen - 2) + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transB'])

        # VIC: We don't want to deal with intermediate results between kernel calls
        tilerModel.addConstraint(ASecondDimVar == parseDict['N'])
        tilerModel.addConstraint(BFirstDimVar == parseDict['N'])

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

        buffA = ctxt.lookup(operatorRepresentation['A'])
        buffB = ctxt.lookup(operatorRepresentation['B'])

        NSize = buffA.shape[-1]
        NOffset = 0

        inputACubes = []
        inputBCubes = []

        replacements = {"M": [], "O": [], "batch": []}

        # Every output is constructed by a pair of inputs. Reconstruct this pair.
        for cube in outputCubes:
            MOffset, OOffset = cube.offset[-2:]
            MSize, OSize = cube.dims[-2:]

            if len(cube.offset) > 2:
                BatchSize = math.prod(cube.dims[:-2])

                if len(cube.offset) > 3:
                    assert all(off == 0 for off in cube.offset[:-3]), (
                        f"Unsupported tiling across leading batch dims: offsets={cube.offset}. "
                        "Only the last batch dim (besides M/O) may be tiled.")
            else:
                BatchSize = 1

            replacements["M"].append(MSize)
            replacements["O"].append(OSize)
            replacements["batch"].append(BatchSize)

            AMatrixOffsets = (MOffset, NOffset)
            AMatrixShape = (MSize, NSize)

            if len(buffA.shape) > 2:
                batchDimCount = len(buffA.shape) - 2
                AMatrixOffsets = tuple(cube.offset[:-2][-batchDimCount:]) + AMatrixOffsets
                AMatrixShape = tuple(cube.dims[:-2][-batchDimCount:]) + AMatrixShape

            ACube = HyperRectangle(AMatrixOffsets, AMatrixShape)
            inputACubes.append(ACube)

            BMatrixOffsets = (NOffset, OOffset)
            BMatrixShape = (NSize, OSize)

            if len(buffB.shape) > 2:
                batchDimCount = len(buffB.shape) - 2
                BMatrixOffsets = tuple(cube.offset[:-2][-batchDimCount:]) + BMatrixOffsets
                BMatrixShape = tuple(cube.dims[:-2][-batchDimCount:]) + BMatrixShape

            BCube = HyperRectangle(BMatrixOffsets, BMatrixShape)
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
