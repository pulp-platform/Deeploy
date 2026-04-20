# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""LayerNorm tile constraint for XDNA2.

The XDNA2 LayerNorm kernel processes one row at a time (``cols`` elements).
The last dimension must remain untiled; outer dimensions can be tiled freely.

Weight and bias are 1-D tensors matching the last dimension.  They are
included in the tiling model for the solver but are NOT streamed via
ObjectFifos (the kernel hardcodes gamma=1, beta=0).
"""

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme

from Deeploy.Targets.XDNA2.TileConstraints.DivisibilityHelper import addDivisibilityConstraints


class XDNA2LayerNormTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']
        scaleBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        for bufferName in [inputBufferName, outputBufferName, scaleBufferName, biasBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputBufferName).shape
        lastDimIdx = len(inputShape) - 1
        lastDimLen = inputShape[-1]

        # Last dimension must remain untiled (kernel processes full rows)
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == lastDimLen)

        # Constrain tile to exactly one row: the kernel normalises a single row
        # per invocation (cols elements).  Multi-row tiles cause corrupted output
        # on AIE2 hardware (llvm-aie codegen issue with row loops).
        for dimIdx in range(lastDimIdx):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = dimIdx) == 1)

        # Scale and bias are 1-D, matching the last dimension
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = scaleBufferName, dimIdx = 0))
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = biasBufferName, dimIdx = 0))

        # Input and output shapes must match
        for idx in range(len(inputShape)):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = outputBufferName, dimIdx = idx))

        # XDNA2: no remainder tiles — every dim must evenly divide the full shape
        addDivisibilityConstraints(tilerModel, outputBufferName, ctxt)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = ['data_in', 'data_out', 'weight', 'bias']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": [], "lastDimLength": []}
        replacementTypes = {"size": PointerClass(uint16_t), "lastDimLength": PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            newSize = int(np.prod(cube.dims))
            replacements["size"].append(newSize)
            replacements["lastDimLength"].append(int(cube.dims[-1]))
            weightCube = HyperRectangle((0,), (cube.dims[-1],))
            biasCube = HyperRectangle((0,), (cube.dims[-1],))
            inputLoadSchedule.append({"data_in": cube, "weight": weightCube, "bias": biasCube})
            outputLoadSchedule.append({"data_out": cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
