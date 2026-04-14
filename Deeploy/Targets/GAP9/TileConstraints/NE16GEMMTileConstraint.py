# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class NE16GEMMTileConstraint(TileConstraint):
    """Tile constraint for NE16 GEMM with bitplane-packed weights stored as 2D [Ko, Ki]."""

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])
        bufferC = ctxt.lookup(name = parseDict['C'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        bufferNames = [bufferA.name, bufferB.name, bufferC.name, outputBuffer.name]
        hasMul = 'mul' in parseDict and isinstance(parseDict['mul'], str)
        if hasMul:
            mulBuffer = ctxt.lookup(name = parseDict['mul'])
            bufferNames.append(mulBuffer.name)
        hasScaleN = 'scale_n' in parseDict and isinstance(parseDict['scale_n'], str)
        if hasScaleN:
            scaleNBuffer = ctxt.lookup(name = parseDict['scale_n'])
            bufferNames.append(scaleNBuffer.name)

        for bufferName in bufferNames:
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

        tilerModel.addConstraint(outputFirstDimVar == AFirstDimVar)
        tilerModel.addConstraint(outputSecondDimVar == BSecondDimVar)
        tilerModel.addConstraint(ASecondDimVar == BFirstDimVar)

        addDimVar = tilerModel.getTensorDimVar(tensorName = bufferC.name, dimIdx = 0)
        tilerModel.addConstraint(outputSecondDimVar == addDimVar)

        if hasMul:
            mulDimVar = tilerModel.getTensorDimVar(tensorName = mulBuffer.name, dimIdx = 0)
            tilerModel.addConstraint(outputSecondDimVar == mulDimVar)

        if hasScaleN:
            scaleNDimVar = tilerModel.getTensorDimVar(tensorName = scaleNBuffer.name, dimIdx = 0)
            tilerModel.addConstraint(outputSecondDimVar == scaleNDimVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])

        dimOffsetA = len(bufferA.shape) - 2
        dimOffsetB = len(bufferB.shape) - 2

        # Don't tile N (reduction dimension) — NE16 needs full input channels
        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = dimOffsetA + 1 - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name, dimIdx = dimOffsetB + parseDict['transB'])
        tilerModel.addConstraint(ASecondDimVar == parseDict['N'])
        tilerModel.addConstraint(BFirstDimVar == parseDict['N'])

        # O (output channels) should be divisible by 32 (NE16 TP_OUT)
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = dimOffsetB + 1 - parseDict['transB'])
        if parseDict["O"] > 32:
            tilerModel.addTileSizeDivisibleConstraint(parseDict, 'O', BSecondDimVar, 32,
                                                       strategy = PerformanceHint(priority = 1))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        hasMul = 'mul' in operatorRepresentation and isinstance(operatorRepresentation['mul'], str)
        hasScaleN = 'scale_n' in operatorRepresentation and isinstance(operatorRepresentation['scale_n'], str)
        addrNames = ['A', 'B', 'C', 'data_out']
        if hasMul:
            addrNames.insert(2, 'mul')
        if hasScaleN:
            addrNames.insert(-1, 'scale_n')
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)
        transA = operatorRepresentation['transA']
        transB = operatorRepresentation['transB']

        buffA = ctxt.lookup(operatorRepresentation['A'])
        buffB = ctxt.lookup(operatorRepresentation['B'])

        NSize = buffA.shape[-1]

        inputACubes = []
        inputBCubes = []
        inputMulCubes = []
        inputAddCubes = []

        replacements = {"M": [], "O": [], "batch": []}

        for cube in outputCubes:
            MOffset, OOffset = cube.offset[-2:]
            MSize, OSize = cube.dims[-2:]

            if len(cube.offset) > 2:
                BatchSize = math.prod(cube.dims[:-2])
            else:
                BatchSize = 1

            replacements["M"].append(MSize)
            replacements["O"].append(OSize)
            replacements["batch"].append(BatchSize)

            if transA == 0:
                AMatrixOffsets = (MOffset, 0)
                AMatrixShape = (MSize, NSize)
            else:
                AMatrixOffsets = (0, MOffset)
                AMatrixShape = (NSize, MSize)

            if len(buffA.shape) > 2:
                batchDimCount = len(buffA.shape) - 2
                AMatrixOffsets = tuple(cube.offset[:-2][-batchDimCount:]) + AMatrixOffsets
                AMatrixShape = tuple(cube.dims[:-2][-batchDimCount:]) + AMatrixShape

            inputACubes.append(HyperRectangle(AMatrixOffsets, AMatrixShape))

            if transB == 0:
                BMatrixOffsets = (0, OOffset)
                BMatrixShape = (NSize, OSize)
            else:
                BMatrixOffsets = (OOffset, 0)
                BMatrixShape = (OSize, NSize)

            inputBCubes.append(HyperRectangle(BMatrixOffsets, BMatrixShape))

            RequantCube = HyperRectangle((OOffset,), (OSize,))
            inputMulCubes.append(RequantCube)
            inputAddCubes.append(RequantCube)

        replacements["N"] = [NSize] * len(outputCubes)

        replacementTypes = {
            "M": PointerClass(uint16_t),
            "N": PointerClass(uint16_t),
            "O": PointerClass(uint16_t),
            "batch": PointerClass(uint8_t)
        }

        inputLoadSchedule = []
        outputLoadSchedule = []

        for idx, (a, b, c) in enumerate(zip(inputACubes, inputBCubes, inputAddCubes)):
            load = {"A": a, "B": b, "C": c}
            if hasMul:
                load["mul"] = inputMulCubes[idx]
            if hasScaleN:
                load["scale_n"] = inputMulCubes[idx]  # same per-channel slice as mul/C
            inputLoadSchedule.append(load)

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule
