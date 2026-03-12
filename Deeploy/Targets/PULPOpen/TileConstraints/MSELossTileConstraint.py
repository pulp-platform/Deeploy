# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme
from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint


class MSELossTileConstraint(TileConstraint):
    """TileConstraint for MSELoss(pred, target) -> scalar loss.

    MSELoss = mean((pred - target)^2) is a global reduction; it cannot be
    meaningfully tiled because the normaliser (N) changes with tile size.
    All input dimensions are pinned to their full size.

    The output is a 0-d scalar represented as a 1-element DMA transfer.
    wrapTilingSolution is overridden to bypass the base-class cube logic which
    cannot handle 0-d shape tensors.
    """

    dataIn1Name = 'pred'
    dataIn2Name = 'target'
    dataOutName = 'loss'

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> TilerModel:
        predName = parseDict[cls.dataIn1Name]
        targetName = parseDict[cls.dataIn2Name]
        # Don't add the scalar loss to the tilerModel — it has 0 dims.
        for bufferName in [predName, targetName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)
        predShape = ctxt.lookup(predName).shape
        for dim in range(len(predShape)):
            predDim = tilerModel.getTensorDimVar(predName, dim)
            targetDim = tilerModel.getTensorDimVar(targetName, dim)
            tilerModel.addConstraint(predDim == targetDim)
        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict,
                            ctxt: NetworkContext) -> TilerModel:
        # Pin every dimension to its full size: MSELoss is a global reduction.
        predName = parseDict[cls.dataIn1Name]
        predBuffer = ctxt.lookup(predName)
        for dimIdx, dimLen in enumerate(predBuffer.shape):
            dimVar = tilerModel.getTensorDimVar(predName, dimIdx)
            tilerModel.addConstraint(dimVar == dimLen)
        return tilerModel

    @classmethod
    def constructSymbolicNodeRep(cls, tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict:
        # num_elements is constant (all dims pinned to full size).
        return parseDict.copy()

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint,
            absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        addrNames = [cls.dataIn1Name, cls.dataIn2Name]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames)

        # Add scalar loss output address if available at this memory level.
        lossName = operatorRepresentation[cls.dataOutName]
        lossAddr = cls.getBaseAddr(tilingSolution, targetMemLevel, lossName)
        if lossAddr != [None]:
            outputBaseOffsets[cls.dataOutName] = lossAddr

        # Load the full pred / target tensors in one DMA.
        predName = operatorRepresentation[cls.dataIn1Name]
        predBuffer = ctxt.lookup(predName)
        fullCube = HyperRectangle((0,) * len(predBuffer.shape), tuple(predBuffer.shape))

        num_elements = operatorRepresentation['num_elements']
        replacements = {'num_elements': [num_elements]}
        replacementTypes = {'num_elements': PointerClass(uint16_t)}

        inputLoadSchedule = [{cls.dataIn1Name: fullCube, cls.dataIn2Name: fullCube}]
        lossRect = HyperRectangle((0,), (1,))
        outputLoadSchedule = [{cls.dataOutName: lossRect}]

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets,
                                        inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)
        return variableReplacementSchedule, tilingSchedule

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str,
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:
        # The single output is a 0-d scalar.  The base-class wrapTilingSolution
        # would crash building HyperRectangle(shape=[]).  We bypass it and call
        # serializeTilingSolution directly with a dummy 1-element scalar rect.
        scalarRect = AbsoluteHyperRectangle(HyperRectangle((0,), (1,)), (0,))
        varReplacement, tilingSchedule = cls.serializeTilingSolution(
            tilingSolution, [scalarRect], targetMemLevel, ctxt, operatorRepresentation)
        tilingSchedule = cls.sanitizeTilingSchedule(tilingSchedule)
        return varReplacement, [tilingSchedule]


class MSELossGradTileConstraint(BOPTileConstraint):
    """TileConstraint for MSELossGrad(pred, target) -> grad.

    The gradient grad[i] = 2*(pred[i]-target[i])/N is element-wise — the same
    tiling works as any binary element-wise op (BOPTileConstraint).
    We only swap the replacement key from 'size' to 'num_elements' to match
    the variable name used in the MSELossGrad C template.
    """

    dataIn1Name = 'pred'
    dataIn2Name = 'target'
    dataOutName = 'grad'

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint,
            absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        import numpy as np
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = [cls.dataIn1Name, cls.dataIn2Name, cls.dataOutName]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames)

        replacements = {'num_elements': []}
        replacementTypes = {'num_elements': PointerClass(uint16_t)}

        for cube in outputCubes:
            replacements['num_elements'].append(int(np.prod(cube.dims)))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            inputLoadSchedule.append({cls.dataIn1Name: cube, cls.dataIn2Name: cube})
            outputLoadSchedule.append({cls.dataOutName: cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets,
                                        inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)
        return variableReplacementSchedule, tilingSchedule
