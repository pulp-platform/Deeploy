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
        # ===== GET NECESSARY INFORMATION =====
        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        tensorsShapeLenA = len(bufferA.shape)
        tensorsShapeLenB = len(bufferB.shape)
        tensorsShapeLenOutput = len(outputBuffer.shape)

        # ===== ADD I/O DIMS TO MODEL AS VARS =====
        for _buffer in [bufferA, bufferB, outputBuffer]:
            tilerModel.addTensorDimToModel(ctxt, _buffer.name)

        # ===== EXTRACT TENSOR DIMS AS VARS =====
        # *Checks on wether dimesnions are reversed via the transA and transB flags
        #   A dims
        AMatrixFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                        dimIdx = (tensorsShapeLenA - 2) + parseDict['transA'])
        AMatrixSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                         dimIdx = (tensorsShapeLenA - 1) - parseDict['transA'])

        #   B dims
        BMatrixFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                        dimIdx = (tensorsShapeLenB - 2) + parseDict['transB'])
        BMatrixSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                         dimIdx = (tensorsShapeLenB - 1) - parseDict['transB'])

        #   Output dims
        outputMatrixFirstDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name,
                                                             dimIdx = (tensorsShapeLenOutput - 2))
        outputMatrixSecondDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name,
                                                              dimIdx = (tensorsShapeLenOutput - 1))

        # ===== ADD CONSTRAINTS =====
        #   Add batch constraints
        if (bufferA.shape[:-2] == bufferB.shape[:-2]):
            for idx in range(tensorsShapeLenA - 2):
                tilerModel.addConstraint(
                    tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = tensorsShapeLenOutput - 3 - idx)
                    == tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = tensorsShapeLenA - 3 - idx))

            for idx in range(tensorsShapeLenB - 2):
                tilerModel.addConstraint(
                    tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = tensorsShapeLenOutput - 3 - idx)
                    == tilerModel.getTensorDimVar(tensorName = bufferB.name, dimIdx = tensorsShapeLenB - 3 - idx))

        #   Add GEMM geometrical constraints
        tilerModel.addConstraint(outputMatrixFirstDimVar == AMatrixFirstDimVar)
        tilerModel.addConstraint(outputMatrixSecondDimVar == BMatrixSecondDimVar)

        tilerModel.addConstraint(AMatrixSecondDimVar == BMatrixFirstDimVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # ===== GET NECESSARY INFORMATION =====
        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])

        # ===== EXTRACT TENSOR DIMS AS VARS =====
        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = (len(bufferA.shape) - 1) - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                  dimIdx = (len(bufferB.shape) - 2) + parseDict['transB'])

        # ===== ADD CONSTRAINTS =====
        # VIC: We don't want to deal with intermediate results between kernel calls
        tilerModel.addConstraint(ASecondDimVar == parseDict['N'])
        tilerModel.addConstraint(BFirstDimVar == parseDict['N'])

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        # Get output cubes
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Get names, optimizer variables, buffers, and other information for elements of interest
        addrNames = ['A', 'B', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        buffA = ctxt.lookup(operatorRepresentation['A'])
        buffB = ctxt.lookup(operatorRepresentation['B'])
        buffOut = ctxt.lookup(operatorRepresentation['data_out'])

        transA = operatorRepresentation['transA']
        transB = operatorRepresentation['transB']

        tensorsShapeLenA = len(buffA.shape)
        tensorsShapeLenB = len(buffB.shape)
        tensorsShapeOutput = len(buffOut.shape)

        # NSize depends on transA: if transA=0, N is last dim; if transA=1, N is second-to-last
        NSize = buffA.shape[-1] if transA == 0 else buffA.shape[-2]
        NOffset = 0

        # Prepare input cubes lists
        inputACubes = []
        inputBCubes = []

        # Prepare replacements lists
        replacements = {"M": [], "O": [], "batch": []}

        # Every output tile is constructed by a pair of input tiles. Reconstruct this pair.
        for cube in outputCubes:
            # Get output dimensions
            MOffset, OOffset = cube.offset[-2:]
            MSize, OSize = cube.dims[-2:]

            # Check that batch tiling is set up properly
            if len(cube.offset) > 2:
                BatchSize = math.prod(cube.dims[:-2])

                if len(cube.offset) > 3:
                    assert all(off == 0 for off in cube.offset[:-3]), (
                        f"Unsupported tiling across leading batch dims: offsets={cube.offset}. "
                        "Only the last batch dim (besides M/O) may be tiled.")
            else:
                BatchSize = 1

            # Prepare cube dimensions replacements
            replacements["M"].append(MSize)
            replacements["O"].append(OSize)
            replacements["batch"].append(BatchSize)

            # ===== Compute A cube information =====
            #   Matrix offsets and shape (swap based on transA)
            if transA == 0:
                AMatrixOffsets = (MOffset, NOffset)
                AMatrixShape = (MSize, NSize)
            else:
                AMatrixOffsets = (NOffset, MOffset)
                AMatrixShape = (NSize, MSize)

            #   Batch offset and shape (with broadcasting handling)
            ABatchOffsets = list()
            ABatchShape = list()

            for idx in range(tensorsShapeLenA - 2):
                if buffA.shape[tensorsShapeLenA - 3 - idx] == buffOut.shape[tensorsShapeOutput - 3 - idx]:
                    ABatchOffsets.append(cube.offset[len(cube.offset) - 3 - idx])
                    ABatchShape.append(cube.dims[len(cube.dims) - 3 - idx])
                else:
                    ABatchOffsets.append(0)
                    ABatchShape.append(1)

            ACube = HyperRectangle(
                tuple(reversed(ABatchOffsets)) + tuple(AMatrixOffsets),
                tuple(reversed(ABatchShape)) + tuple(AMatrixShape))
            inputACubes.append(ACube)

            # ===== Compute B cube information =====
            #   Matrix offsets and shape (swap based on transB)
            if transB == 0:
                BMatrixOffsets = (NOffset, OOffset)
                BMatrixShape = (NSize, OSize)
            else:
                BMatrixOffsets = (OOffset, NOffset)
                BMatrixShape = (OSize, NSize)

            #   Batch offset and shape (with broadcasting handling)
            BBatchOffsets = list()
            BBatchShape = list()

            for idx in range(tensorsShapeLenB - 2):
                if buffB.shape[tensorsShapeLenB - 3 - idx] == buffOut.shape[tensorsShapeOutput - 3 - idx]:
                    BBatchOffsets.append(cube.offset[len(cube.offset) - 3 - idx])
                    BBatchShape.append(cube.dims[len(cube.dims) - 3 - idx])
                else:
                    BBatchOffsets.append(0)
                    BBatchShape.append(1)

            BCube = HyperRectangle(
                tuple(reversed(BBatchOffsets)) + tuple(BMatrixOffsets),
                tuple(reversed(BBatchShape)) + tuple(BMatrixShape))
            inputBCubes.append(BCube)

        # Prepare load schedule lists for computed cubes
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Prepare replacements
        replacements["N"] = [NSize] * len(outputCubes)

        replacementTypes = {
            "M": PointerClass(int8_t),
            "N": PointerClass(int8_t),
            "O": PointerClass(int8_t),
            "batch": PointerClass(int8_t)
        }

        # Update load schedule lists
        # *With strict=True to fail fast if different list lenghts
        for a, b in zip(inputACubes, inputBCubes, strict = True):
            inputLoadSchedule.append({"A": a, "B": b})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        # Prepare tiling schedule object
        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule
