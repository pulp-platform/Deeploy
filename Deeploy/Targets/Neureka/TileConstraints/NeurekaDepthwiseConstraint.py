# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.Neureka.Templates.ConvTemplate import Neureka2DDWConvTemplate
from Deeploy.Targets.Neureka.TileConstraints.NeurekaConvTileConstraint import NeurekaConvTileConstraint, \
    NeurekaRQSConvTileConstraint, PerTileReplacements
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, calculateFlatOffsetInBytes

# Neureka packs depthwise weights into "cinMajor" blocks of this many channels (see
# NeurekaAdjustWeightMemoryLayoutPass). A channel tile can therefore only start on a boundary that
# is a multiple of this value.
_NEUREKA_CIN_SUBTILE_3x3 = 28
_NEUREKA_KERNEL_HEIGHT_3x3 = 3
_NEUREKA_KERNEL_WIDTH_3x3 = 3


class NeurekaDWConv2DTileConstraint(NeurekaConvTileConstraint):

    _ConvTemplate = Neureka2DDWConvTemplate

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        outputBufferName = parseDict['data_out']

        strides = parseDict["strides"]
        pads = parseDict["pads"]

        for bufferName in [inputBufferName, weightBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        # In depthwise this axis is degenerate (cout == 1): the actual channels are folded into `cinMajor`,
        # not into cout. it is just the size-1 output-channel axis of the weight blob.
        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)

        tilerModel.addConstraint(weightOutChannelVar == weightOutChannelVar.Max())  # dummy since cout=1

        # Since channels are packed in blocks of _NEUREKA_CIN_SUBTILE_3x3 channels, either
        # - channels are not tiled (single tile == full size) or
        # - channels are tiles with a tile size multiple of _NEUREKA_CIN_SUBTILE_3x3
        tilerModel.addConstraint((outputChannelVar == outputChannelVar.Max()) +
                                 ((outputChannelVar % _NEUREKA_CIN_SUBTILE_3x3) == 0) >= 1)

        tilerModel.addConstraint(inputHeightVar >= _NEUREKA_KERNEL_HEIGHT_3x3)
        tilerModel.addConstraint(inputWidthVar >= _NEUREKA_KERNEL_WIDTH_3x3)

        _, Hin, Win, _ = ctxt.lookup(inputBufferName).shape
        effectiveHeight = inputHeightVar + ((pads[0] + pads[2]) * (inputHeightVar == Hin))
        effectiveWidth = inputWidthVar + ((pads[1] + pads[3]) * (inputWidthVar == Win))
        outputHeight = (effectiveHeight - (_NEUREKA_KERNEL_HEIGHT_3x3 - 1) - 1) // strides[0] + 1
        outputWidth = (effectiveWidth - (_NEUREKA_KERNEL_WIDTH_3x3 - 1) - 1) // strides[1] + 1
        tilerModel.addConstraint(outputHeightVar == outputHeight)
        tilerModel.addConstraint(outputWidthVar == outputWidth)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = parseDict['data_in'], dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = parseDict['data_in'], dimIdx = 2)

        weightInChannelMajorVar = tilerModel.getTensorDimVar(tensorName = parseDict['weight'], dimIdx = 1)
        weightBitsVar = tilerModel.getTensorDimVar(tensorName = parseDict['weight'], dimIdx = 2)
        weightBandwidthVar = tilerModel.getTensorDimVar(tensorName = parseDict['weight'], dimIdx = 3)

        strides = parseDict["strides"]

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        # Force the weight tensor's non-tiled dims to their full size
        tilerModel.addConstraint(weightInChannelMajorVar == weightInChannelMajorVar.Max())
        tilerModel.addConstraint(weightBitsVar == weightBitsVar.Max())
        tilerModel.addConstraint(weightBandwidthVar == weightBandwidthVar.Max())

        tilerModel.addConstraint(inputHeightVar == inputHeightVar.Max(), strategy = PerformanceHint(1))
        tilerModel.addConstraint(inputWidthVar == inputWidthVar.Max(), strategy = PerformanceHint(1))

        return tilerModel

    @classmethod
    def _adjustInputCube(cls, inCube: HyperRectangle, outputCube: HyperRectangle) -> HyperRectangle:
        # In DW, each output channel only depends on the corresponding input channel.
        # Therefore we can tile the input channels exactly as the output channels.
        COffset = outputCube.offset[-1]
        CSize = outputCube.dims[-1]
        return HyperRectangle(inCube.offset[:-1] + (COffset,), inCube.dims[:-1] + (CSize,))

    @classmethod
    def _addWeightSchedule(cls, rep: PerTileReplacements, inputLoadSchedule: List[Dict[str, HyperRectangle]],
                           inputBaseOffsets: Dict[str, List[int]], outputBaseOffsets: Dict[str, List[int]],
                           absoluteOutputCubes: List[AbsoluteHyperRectangle], tilingSolution: NodeMemoryConstraint,
                           targetMemLevel: str, ctxt: NetworkContext,
                           operatorRepresentation: OperatorRepresentation) -> None:
        weightBuffer = ctxt.lookup(operatorRepresentation['weight'])
        assert isinstance(weightBuffer, VariableBuffer)
        weightShape = weightBuffer.shape

        # The DW weight is never tiled: it is always resident in full (in SRAM for the wmem case, or DMA'd
        # whole into L1 otherwise). It is packed as (cout=1, cinMajor, bits, bandwidthBytes), where the
        # channels live in the cinMajor dimension in blocks of _NEUREKA_CIN_SUBTILE_3x3. A channel tile
        # starting at COffset (guaranteed to be a multiple of _NEUREKA_CIN_SUBTILE_3x3 by the geometrical
        # constraint) therefore starts at cinMajor block COffset // _NEUREKA_CIN_SUBTILE_3x3, so we offset
        # the weight base to that block.
        for absoluteCube in absoluteOutputCubes:
            COffset = absoluteCube.absoluteOffset[-1]
            cinMajorOffset = COffset // _NEUREKA_CIN_SUBTILE_3x3
            WeightCube = HyperRectangle((0, cinMajorOffset, 0, 0),
                                        (weightShape[0], 1, weightShape[-2], weightShape[-1]))
            rep.append('weight_addr_offset', uint32_t, calculateFlatOffsetInBytes(WeightCube, weightBuffer))

        if not (hasattr(weightBuffer, "_memoryLevel") and weightBuffer._memoryLevel == "WeightMemory_SRAM"):
            inputWeightBaseOffsets, outputWeightBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                                  operatorRepresentation, ['weight'])
            inputBaseOffsets.update(inputWeightBaseOffsets)
            outputBaseOffsets.update(outputWeightBaseOffsets)

            for load in inputLoadSchedule:
                load['weight'] = HyperRectangle((0,) * len(weightShape), tuple(weightShape))


class NeurekaRQSDWConv2DTileConstraint(NeurekaRQSConvTileConstraint, NeurekaDWConv2DTileConstraint):
    pass
