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

import math
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
        """
            Serialize a tiling solution for a GEMM-like operator into a VariableReplacementScheme and a TilingSchedule.
            
            Constructs input/output HyperRectangle tiles and replacement vectors from the provided absolute output cubes and the operator representation:
            - Treats the last two tensor dimensions of each output cube as M and O respectively; any leading dimensions are treated as batch dimensions and their product becomes the `batch` replacement.
            - Reconstructs corresponding A and B input tiles (respecting transposition flags `transA` / `transB`) and a per-output requantization tile for the O dimension.
            - If an input buffer has more than two shape dimensions, the corresponding leading batch offsets/shapes from the output cube are prepended to the input tile offsets/shapes.
            - Populates replacement lists for M, N, O, and batch and assigns pointer-sized replacement types (uint16 for M/N/O, uint8 for batch).
            - Asserts that when there are more than three offset entries for an output cube, any extra leading offsets (above the batch dimensions considered) are zero (sanity check that upper dimensions are not tiled).
            
            Parameters:
            - tilingSolution: NodeMemoryConstraint describing the tiling solution used to extract base addresses.
            - absoluteOutputCubes: list of AbsoluteHyperRectangle objects representing absolute output rectangles to serialize.
            - targetMemLevel: memory level name used when extracting base addresses.
            - ctxt: network context used to look up buffer metadata (e.g., shapes) — treated as a contextual service and not documented in detail.
            - operatorRepresentation: mapping with operator attributes required for serialization (must include keys 'A', 'B', 'transA', 'transB', and 'mul'/'C'/'data_out' names referenced by addrNames).
            
            Returns:
            A tuple (VariableReplacementScheme, TilingSchedule):
            - VariableReplacementScheme: contains replacement value lists for keys "M", "N", "O", and "batch" and their pointer-sized types.
            - TilingSchedule: contains base addresses and per-tile input/output load schedules built from the reconstructed HyperRectangles.
            """
            outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['A', 'B', 'mul', 'C', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)
        transA = operatorRepresentation['transA']
        transB = operatorRepresentation['transB']

        buffA = ctxt.lookup(operatorRepresentation['A'])
        buffB = ctxt.lookup(operatorRepresentation['B'])

        NSize = buffA.shape[-1]
        NOffset = 0

        inputACubes = []
        inputBCubes = []
        inputMulCubes = []
        inputAddCubes = []

        replacements = {"M": [], "O": [], "batch": []}

        # Every output is constructed by a pair of inputs. Reconstruct this pair.
        for cube in outputCubes:
            MOffset, OOffset = cube.offset[-2:]
            MSize, OSize = cube.dims[-2:]

            if len(cube.offset) > 2:
                BatchSize = math.prod(cube.dims[:-2])

                # Check that we don't tile upper dimensions
                if len(cube.offset) > 3:
                    assert all(off == 0 for off in cube.offset[:-3])
            else:
                BatchSize = 1

            replacements["M"].append(MSize)
            replacements["O"].append(OSize)
            replacements["batch"].append(BatchSize)

            if transA == 0:
                AMatrixOffsets = (MOffset, NOffset)
                AMatrixShape = (MSize, NSize)
            else:
                AMatrixOffsets = (NOffset, MOffset)
                AMatrixShape = (NSize, MSize)

            if len(buffA.shape) > 2:
                batchDimCount = len(buffA.shape) - 2
                AMatrixOffsets = tuple(cube.offset[:-2][-batchDimCount:]) + AMatrixOffsets
                AMatrixShape = tuple(cube.dims[:-2][-batchDimCount:]) + AMatrixShape

            ACube = HyperRectangle(AMatrixOffsets, AMatrixShape)

            if transB == 0:
                BMatrixOffsets = (NOffset, OOffset)
                BMatrixShape = (NSize, OSize)
            else:
                BMatrixOffsets = (OOffset, NOffset)
                BMatrixShape = (OSize, NSize)

            if len(buffB.shape) > 2:
                batchDimCount = len(buffB.shape) - 2
                BMatrixOffsets = tuple(cube.offset[:-2][-batchDimCount:]) + BMatrixOffsets
                BMatrixShape = tuple(cube.dims[:-2][-batchDimCount:]) + BMatrixShape

            BCube = HyperRectangle(BMatrixOffsets, BMatrixShape)

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
        bufferC = ctxt.lookup(name = parseDict['C'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        # Add I/O dimensions to the model as variables
        for bufferName in [bufferA.name, bufferB.name, bufferC.name, outputBuffer.name]:
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

        addDimVar_1 = tilerModel.getTensorDimVar(tensorName = bufferC.name, dimIdx = 0)
        addDimVar_2 = tilerModel.getTensorDimVar(tensorName = bufferC.name, dimIdx = 1)
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
        """
            Build a VariableReplacementScheme and TilingSchedule by reconstructing input tiles (A, B, C) from absolute output cubes for a GEMM-like operator.
            
            Detailed behavior:
            - Uses cls.extractBaseAddr to compute base offsets for inputs ['A','B','C','data_out'] at the given memory level.
            - For each AbsoluteHyperRectangle in absoluteOutputCubes, treats the last two dimensions as (M, O) and any leading dimensions as batch dimensions. BatchSize is the product of leading dims when present; if there are more than three offset entries the function asserts that all higher offsets are zero.
            - Reconstructs HyperRectangles for A, B, and C respecting transposition flags operatorRepresentation['transA'] and operatorRepresentation['transB'], and prepends any required batch-dimension offsets/shapes when the corresponding buffer has more than two shape dimensions.
            - Produces replacement lists for variables "M", "O", "batch", and a constant "N" derived from the A buffer shape (respecting transposition). Replacement pointer types are set for M, N, O, and batch.
            - Assembles inputLoadSchedule entries pairing reconstructed A/B/C tiles and outputLoadSchedule entries for data_out, then returns the VariableReplacementScheme and the constructed TilingSchedule.
            
            Parameters:
                absoluteOutputCubes (List[AbsoluteHyperRectangle]): output rectangles from the tiling solution; last two axes are interpreted as M and O.
                operatorRepresentation (OperatorRepresentation): mapping that must include keys 'A', 'B', 'C', 'transA', and 'transB' used to locate buffers and determine transposition.
            
            Returns:
                Tuple[VariableReplacementScheme, TilingSchedule]: a replacement scheme mapping variables ("M","N","O","batch") to lists and types, and a TilingSchedule containing base offsets and per-tile load schedules.
            """
            outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['A', 'B', 'C', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        transA = operatorRepresentation['transA']
        transB = operatorRepresentation['transB']

        buffA = ctxt.lookup(operatorRepresentation['A'])
        buffB = ctxt.lookup(operatorRepresentation['B'])
        buffC = ctxt.lookup(operatorRepresentation['C'])

        if transA == 0:
            NSize = buffA.shape[-1]
        else:
            NSize = buffA.shape[-2]

        NOffset = 0

        inputACubes = []
        inputBCubes = []
        inputAddCubes = []

        replacements = {"M": [], "O": [], "batch": []}

        # Every output is constructed by a pair of inputs. Reconstruct this pair.
        for cube in outputCubes:
            MOffset, OOffset = cube.offset[-2:]
            MSize, OSize = cube.dims[-2:]

            if len(cube.offset) > 2:
                BatchSize = math.prod(cube.dims[:-2])

                # Check that we don't tile upper dimensions
                if len(cube.offset) > 3:
                    assert all(off == 0 for off in cube.offset[:-3])
            else:
                BatchSize = 1

            replacements["M"].append(MSize)
            replacements["O"].append(OSize)
            replacements["batch"].append(BatchSize)

            if transA == 0:
                AMatrixOffsets = (MOffset, NOffset)
                AMatrixShape = (MSize, NSize)
            else:
                AMatrixOffsets = (NOffset, MOffset)
                AMatrixShape = (NSize, MSize)

            if len(buffA.shape) > 2:
                batchDimCount = len(buffA.shape) - 2
                AMatrixOffsets = tuple(cube.offset[:-2][-batchDimCount:]) + AMatrixOffsets
                AMatrixShape = tuple(cube.dims[:-2][-batchDimCount:]) + AMatrixShape

            ACube = HyperRectangle(AMatrixOffsets, AMatrixShape)

            if transB == 0:
                BMatrixOffsets = (NOffset, OOffset)
                BMatrixShape = (NSize, OSize)
            else:
                BMatrixOffsets = (OOffset, NOffset)
                BMatrixShape = (OSize, NSize)

            if len(buffB.shape) > 2:
                batchDimCount = len(buffB.shape) - 2
                BMatrixOffsets = tuple(cube.offset[:-2][-batchDimCount:]) + BMatrixOffsets
                BMatrixShape = tuple(cube.dims[:-2][-batchDimCount:]) + BMatrixShape

            BCube = HyperRectangle(BMatrixOffsets, BMatrixShape)

            CMatrixOffsets = (MOffset, OOffset)
            CMatrixShape = (MSize, OSize)

            if len(buffC.shape) > 2:
                batchDimCount = len(buffC.shape) - 2
                CMatrixOffsets = tuple(cube.offset[:-2][-batchDimCount:]) + CMatrixOffsets
                CMatrixShape = tuple(cube.dims[:-2][-batchDimCount:]) + CMatrixShape

            CCube = HyperRectangle(CMatrixOffsets, CMatrixShape)

            inputACubes.append(ACube)
            inputBCubes.append(BCube)
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

        for a, b, c in zip(inputACubes, inputBCubes, inputAddCubes):
            inputLoadSchedule.append({"A": a, "B": b, "C": c})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule
