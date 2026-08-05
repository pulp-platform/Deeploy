# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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


class Pad2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        buffersOfInterest = [inputBufferName, outputBufferName]
        if 'pads_tensor' in parseDict:
            buffersOfInterest.append(parseDict['pads_tensor'])
        if 'value_tensor' in parseDict:
            buffersOfInterest.append(parseDict['value_tensor'])

        for bufferName in buffersOfInterest:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        channels_first = bool(parseDict.get('channels_first', 1))

        if channels_first:
            batchDim = 0
            channelDim = 1
            heightDim = 2
            widthDim = 3
        else:
            batchDim = 0
            heightDim = 1
            widthDim = 2
            channelDim = 3

        inputBatchVar = tilerModel.getTensorDimVar(inputBufferName, batchDim)
        inputHeightVar = tilerModel.getTensorDimVar(inputBufferName, heightDim)
        inputWidthVar = tilerModel.getTensorDimVar(inputBufferName, widthDim)
        inputChannelVar = tilerModel.getTensorDimVar(inputBufferName, channelDim)

        outputBatchVar = tilerModel.getTensorDimVar(outputBufferName, batchDim)
        outputHeightVar = tilerModel.getTensorDimVar(outputBufferName, heightDim)
        outputWidthVar = tilerModel.getTensorDimVar(outputBufferName, widthDim)
        outputChannelVar = tilerModel.getTensorDimVar(outputBufferName, channelDim)

        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputWidthVar == inputWidthVar + 2 * parseDict['pad_x'])
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)

        # Height tiles may include top/bottom padding only at the boundary tiles.
        tilerModel.addConstraint(outputHeightVar >= inputHeightVar)
        tilerModel.addConstraint(outputHeightVar <= inputHeightVar + 2 * parseDict['pad_y'])

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        channels_first = bool(parseDict.get('channels_first', 1))

        if channels_first:
            batchDim = 0
            channelDim = 1
            heightDim = 2
            widthDim = 3
        else:
            batchDim = 0
            heightDim = 1
            widthDim = 2
            channelDim = 3

        inputBuffer = ctxt.lookup(inputBufferName)
        outputBuffer = ctxt.lookup(outputBufferName)

        # Keep batch, width and channels whole to preserve contiguous NHWC row copies.
        tilerModel.addConstraint(tilerModel.getTensorDimVar(inputBufferName, batchDim) == inputBuffer.shape[batchDim])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(outputBufferName, batchDim) == outputBuffer.shape[batchDim])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(inputBufferName, widthDim) == inputBuffer.shape[widthDim])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(outputBufferName, widthDim) == outputBuffer.shape[widthDim])
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(inputBufferName, channelDim) == inputBuffer.shape[channelDim])
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(outputBufferName, channelDim) == outputBuffer.shape[channelDim])

        # Avoid pure-padding tiles and avoid tail tiles with a single padding row.
        outputHeightVar = tilerModel.getTensorDimVar(outputBufferName, heightDim)
        tilerModel.addConstraint(outputHeightVar >= (parseDict['pad_y'] + 1))

        return tilerModel

    @staticmethod
    def _computeInputCube(outputCube: HyperRectangle, parseDict: Dict,
                          outputShape: Tuple[int, ...]) -> Tuple[HyperRectangle, int]:
        channels_first = bool(parseDict.get('channels_first', 1))

        if channels_first:
            batchOffset, _, outputHOffset, _ = outputCube.offset
            batchSize, _, outputHSize, _ = outputCube.dims
            inputChannels = parseDict['dim_im_in_ch']
            inputWidth = parseDict['dim_im_in_y']
        else:
            batchOffset, outputHOffset, _, _ = outputCube.offset
            batchSize, outputHSize, _, _ = outputCube.dims
            inputChannels = parseDict['dim_im_in_ch']
            inputWidth = parseDict['dim_im_in_y']

        padTop = parseDict['pad_y']
        inputHeightTotal = parseDict['dim_im_in_x']
        outputHeightTotal = outputShape[2] if channels_first else outputShape[1]

        dataStart = padTop
        dataEnd = padTop + inputHeightTotal
        tileEnd = outputHOffset + outputHSize

        localPadTop = max(dataStart - outputHOffset, 0)
        localPadBottom = max(tileEnd - dataEnd, 0)

        inputStart = max(outputHOffset, dataStart) - dataStart
        inputEnd = min(tileEnd, dataEnd) - dataStart
        inputHeight = max(inputEnd - inputStart, 1)

        if channels_first:
            inCube = HyperRectangle((batchOffset, 0, inputStart, 0),
                                    (batchSize, inputChannels, inputHeight, inputWidth))
        else:
            inCube = HyperRectangle((batchOffset, inputStart, 0, 0),
                                    (batchSize, inputHeight, inputWidth, inputChannels))

        _ = localPadBottom  # Bottom padding is encoded by the output height itself.
        return inCube, localPadTop

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements: Dict[str, List[int]] = {
            "batch": [],
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_in_ch": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "dim_im_out_ch": [],
            "pad_y": [],
            "data_out_size": [],
        }

        replacementTypes = {
            "batch": PointerClass(uint16_t),
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_in_ch": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "dim_im_out_ch": PointerClass(uint16_t),
            "pad_y": PointerClass(uint16_t),
            "data_out_size": PointerClass(uint16_t),
        }

        outputShape = tuple(ctxt.lookup(operatorRepresentation['data_out']).shape)
        channels_first = bool(operatorRepresentation.get('channels_first', 1))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for outCube in outputCubes:
            inCube, localPadTop = cls._computeInputCube(outCube, operatorRepresentation, outputShape)

            replacements["batch"].append(outCube.dims[0])
            replacements["pad_y"].append(localPadTop)
            replacements["data_out_size"].append(int(np.prod(outCube.dims)))

            if channels_first:
                replacements["dim_im_in_ch"].append(inCube.dims[1])
                replacements["dim_im_in_x"].append(inCube.dims[2])
                replacements["dim_im_in_y"].append(inCube.dims[3])
                replacements["dim_im_out_ch"].append(outCube.dims[1])
                replacements["dim_im_out_x"].append(outCube.dims[2])
                replacements["dim_im_out_y"].append(outCube.dims[3])
            else:
                replacements["dim_im_in_x"].append(inCube.dims[1])
                replacements["dim_im_in_y"].append(inCube.dims[2])
                replacements["dim_im_in_ch"].append(inCube.dims[3])
                replacements["dim_im_out_x"].append(outCube.dims[1])
                replacements["dim_im_out_y"].append(outCube.dims[2])
                replacements["dim_im_out_ch"].append(outCube.dims[3])

            inputLoadSchedule.append({"data_in": inCube})
            outputLoadSchedule.append({"data_out": outCube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
