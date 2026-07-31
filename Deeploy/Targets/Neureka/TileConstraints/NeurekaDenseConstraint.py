# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.Neureka.Templates.ConvTemplate import Neureka2DDenseConvTemplate
from Deeploy.Targets.Neureka.TileConstraints.NeurekaConvTileConstraint import NeurekaConvTileConstraint, \
    NeurekaRQSConvTileConstraint, PerTileReplacements
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, calculateFlatOffsetInBytes


class NeurekaDenseConv2DTileConstraint(NeurekaConvTileConstraint):

    _ConvTemplate = Neureka2DDenseConvTemplate

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        outputBufferName = parseDict['data_out']

        strides = parseDict["strides"]
        padding = parseDict["pads"]
        dilation = parseDict["dilations"]

        for bufferName in [inputBufferName, weightBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightInChannelMajorVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightBitsVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)
        weightBandwidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 3)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)

        weightBuffer = ctxt.lookup(weightBufferName)
        if hasattr(weightBuffer, "_memoryLevel") and weightBuffer._memoryLevel == "WeightMemory_SRAM":
            # No tiling. Weight tensor is a constant statically placed in the weight memory (wmem)
            tilerModel.addConstraint(weightOutChannelVar == weightOutChannelVar.Max())
        else:
            tilerModel.addConstraint(weightOutChannelVar == outputChannelVar)

        inputBuffer = ctxt.lookup(inputBufferName)

        effectiveHeight = inputHeightVar + ((padding[0] + padding[2]) * (inputHeightVar == inputBuffer.shape[1]))
        effectiveWidth = inputWidthVar + ((padding[1] + padding[3]) * (inputWidthVar == inputBuffer.shape[2]))

        tilerModel.addConstraint((outputHeightVar == (effectiveHeight - (3 - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint((outputWidthVar == (effectiveWidth - (3 - 1) - 1) // strides[1] + 1))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = parseDict['data_in'], dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = parseDict['data_in'], dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = parseDict['data_in'], dimIdx = 3)

        weightInChannelMajorVar = tilerModel.getTensorDimVar(tensorName = parseDict['weight'], dimIdx = 1)
        weightBitsVar = tilerModel.getTensorDimVar(tensorName = parseDict['weight'], dimIdx = 2)
        weightBandwidthVar = tilerModel.getTensorDimVar(tensorName = parseDict['weight'], dimIdx = 3)

        strides = parseDict["strides"]

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        tilerModel.addConstraint(inputChannelVar == inputChannelVar.Max())

        # Force the weight tensor's non-tiled dims to their full size
        tilerModel.addConstraint(weightInChannelMajorVar == weightInChannelMajorVar.Max())
        tilerModel.addConstraint(weightBitsVar == weightBitsVar.Max())
        tilerModel.addConstraint(weightBandwidthVar == weightBandwidthVar.Max())

        tilerModel.addConstraint(inputHeightVar == inputHeightVar.Max(), strategy = PerformanceHint(1))
        tilerModel.addConstraint(inputWidthVar == inputWidthVar.Max(), strategy = PerformanceHint(1))

        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(inputWidthVar >= parseDict['dim_kernel_y'])

        return tilerModel

    @classmethod
    def _addWeightSchedule(cls, rep: PerTileReplacements, inputLoadSchedule: List[Dict[str, HyperRectangle]],
                           inputBaseOffsets: Dict[str, List[int]], outputBaseOffsets: Dict[str, List[int]],
                           absoluteOutputCubes: List[AbsoluteHyperRectangle], tilingSolution: NodeMemoryConstraint,
                           targetMemLevel: str, ctxt: NetworkContext,
                           operatorRepresentation: OperatorRepresentation) -> None:

        weightBuffer = ctxt.lookup(operatorRepresentation['weight'])
        assert isinstance(weightBuffer, VariableBuffer)
        weightShape = weightBuffer.shape

        if hasattr(weightBuffer, "_memoryLevel") and weightBuffer._memoryLevel == "WeightMemory_SRAM":
            for absoluteCube in absoluteOutputCubes:
                COffset, CSize = absoluteCube.absoluteOffset[-1], absoluteCube.rectangle.dims[-1]
                WeightCube = HyperRectangle((COffset, 0, 0, 0),
                                            (CSize, weightShape[-3], weightShape[-2], weightShape[-1]))
                rep.append('weight_addr_offset', uint32_t, calculateFlatOffsetInBytes(WeightCube, weightBuffer))
        else:
            inputWeightBaseOffsets, outputWeightBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                                  operatorRepresentation, ['weight'])
            inputBaseOffsets.update(inputWeightBaseOffsets)
            outputBaseOffsets.update(outputWeightBaseOffsets)

            for absoluteCube, load in zip(absoluteOutputCubes, inputLoadSchedule):
                COffset, CSize = absoluteCube.absoluteOffset[-1], absoluteCube.rectangle.dims[-1]
                load['weight'] = HyperRectangle((COffset, 0, 0, 0),
                                                (CSize, weightShape[-3], weightShape[-2], weightShape[-1]))


class NeurekaRQSDenseConv2DTileConstraint(NeurekaRQSConvTileConstraint, NeurekaDenseConv2DTileConstraint):
    pass
