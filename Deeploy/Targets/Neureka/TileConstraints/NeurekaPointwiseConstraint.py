# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.Neureka.Templates.ConvTemplate import Neureka2DPWConvTemplate
from Deeploy.Targets.Neureka.TileConstraints.NeurekaConvTileConstraint import NeurekaConvTileConstraint, \
    NeurekaRQSConvTileConstraint, PerTileReplacements
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, calculateFlatOffsetInBytes

_NEUREKA_PE_H = 6
_NEUREKA_PE_W = 6
_NEUREKA_TP_IN = 32  # input channel parallelism
_NEUREKA_TP_OUT = 32  # output channel parallelism


class NeurekaPWConv2DTileConstraint(NeurekaConvTileConstraint):

    _ConvTemplate = Neureka2DPWConvTemplate

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        outputBufferName = parseDict['data_out']

        for bufferName in [inputBufferName, weightBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputHeightVar == inputHeightVar)
        tilerModel.addConstraint(outputWidthVar == inputWidthVar)

        weightBuffer = ctxt.lookup(weightBufferName)
        if hasattr(weightBuffer, "_memoryLevel") and weightBuffer._memoryLevel == "WeightMemory_SRAM":
            tilerModel.addConstraint(weightOutChannelVar == weightOutChannelVar.Max())
        else:
            tilerModel.addConstraint(weightOutChannelVar == outputChannelVar)

        tilerModel.addConstraint(inputHeightVar >= 1)
        tilerModel.addConstraint(inputWidthVar >= 1)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)
        weightInChannelMajorVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightBandwidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)

        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 3)

        # Neureka has no batch counter: process one batch element per dispatch
        tilerModel.addConstraint(outputBatchVar == 1)

        strides = parseDict["strides"]

        # LMACAN: Force full input channel to avoid partial results
        tilerModel.addConstraint(inputChannelVar == inputChannelVar.Max())
        tilerModel.addConstraint(weightInChannelMajorVar == weightInChannelMajorVar.Max())
        tilerModel.addConstraint(weightBandwidthVar == weightBandwidthVar.Max())

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        # N-EUREKA tile constraints to align with N-EUREKA's hardware subtiling
        if parseDict["dim_im_out_x"] > _NEUREKA_PE_W:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "dim_im_out_x",
                                                      outputHeightVar,
                                                      _NEUREKA_PE_W,
                                                      strategy = PerformanceHint(priority = 3))
        else:
            tilerModel.addConstraint(outputHeightVar == outputHeightVar.Max(), strategy = PerformanceHint(priority = 3))

        if parseDict["dim_im_out_y"] > _NEUREKA_PE_H:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "dim_im_out_y",
                                                      outputWidthVar,
                                                      _NEUREKA_PE_H,
                                                      strategy = PerformanceHint(priority = 2))
        else:
            tilerModel.addConstraint(outputWidthVar == outputWidthVar.Max(), strategy = PerformanceHint(priority = 2))

        if parseDict["ch_im_out"] > _NEUREKA_TP_OUT:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "ch_im_out",
                                                      outputChannelVar,
                                                      _NEUREKA_TP_OUT,
                                                      strategy = PerformanceHint(priority = 1))
        else:
            tilerModel.addConstraint(outputChannelVar == outputChannelVar.Max(),
                                     strategy = PerformanceHint(priority = 1))

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
                WeightCube = HyperRectangle((COffset, 0, 0), (CSize, weightShape[-2], weightShape[-1]))
                rep.append('weight_addr_offset', uint32_t, calculateFlatOffsetInBytes(WeightCube, weightBuffer))
        else:
            inputWeightBaseOffsets, outputWeightBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                                  operatorRepresentation, ['weight'])
            inputBaseOffsets.update(inputWeightBaseOffsets)
            outputBaseOffsets.update(outputWeightBaseOffsets)

            for absoluteCube, load in zip(absoluteOutputCubes, inputLoadSchedule):
                COffset, CSize = absoluteCube.absoluteOffset[-1], absoluteCube.rectangle.dims[-1]
                load['weight'] = HyperRectangle((COffset, 0, 0), (CSize, weightShape[-2], weightShape[-1]))


class NeurekaRQSPWConv2DTileConstraint(NeurekaRQSConvTileConstraint, NeurekaPWConv2DTileConstraint):
    pass
