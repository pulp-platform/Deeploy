# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t, uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.NE16.Templates.ConvTemplate import NE162DDWConvTemplate, getInputAddrOffset, \
    ioStridesFromDimensions
from Deeploy.Targets.NE16.TileConstraints.RequantHelpers import requantAddGeometricalConstraint, requantLoadSchedule
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class NE16DWConv2DTileConstraint(TileConstraint):

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

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)

        weightBuffer = ctxt.lookup(weightBufferName)
        # NE16 DW weight is packed as a single (1, 1, packed_bytes) block
        # containing all output channels (up to NE16_SUBTILE_INPUT_CHANNEL=16).
        # Keep the outermost dim fixed at its full (=1) value regardless of
        # the output channel tiling.
        tilerModel.addConstraint(weightOutChannelVar == weightOutChannelVar.Max())

        # The depthwise weights are bit-serialised by _weightEncode(depthwise=True)
        # into a single packed block that interleaves up to
        # NE16_SUBTILE_INPUT_CHANNEL=16 parallel output channels, and
        # serializeTilingSolution consequently loads the whole block, from offset
        # 0, for every tile. A channel tile therefore only lines up with the
        # filters it is given when it starts at channel 0 -- split 16 channels
        # into 14 + 2 and the second tile computes channels 14..15 using the
        # filters of channels 0..1, which is wholly wrong output. The packed form
        # cannot be sliced at an arbitrary channel offset, so keep the output
        # channels untiled; a layer that does not fit L1 has to be split
        # spatially instead.
        tilerModel.addConstraint(outputChannelVar == outputChannelVar.Max())

        tilerModel.addConstraint(inputHeightVar >= 3)
        tilerModel.addConstraint(inputWidthVar >= 3)

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

        strides = parseDict["strides"]

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        tilerModel.addConstraint(inputHeightVar == inputHeightVar.Max(), strategy = PerformanceHint(1))
        tilerModel.addConstraint(inputWidthVar == inputWidthVar.Max(), strategy = PerformanceHint(1))

        outputHeightVar = tilerModel.getTensorDimVar(tensorName = parseDict['data_out'], dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = parseDict['data_out'], dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = parseDict['data_out'], dimIdx = 3)

        # Align the output-channel tile with NE16's TP_OUT=32 subtiling. The
        # spatial dimensions are deliberately *not* aligned here: measured on the
        # double-buffered DW_2D_RQ kernel, adding a divisible-by-3 constraint on
        # dim_im_out_x/y costs 11,926 -> 19,391 cycles with no measured gain
        # anywhere else, because the halo re-fetch from the extra split outweighs
        # the partially-filled border pass it avoids. Depthwise has no input-
        # channel reuse to amortise that halo against.
        #
        # a body tile that is not a multiple of those leaves part of the array
        # idle. addTileSizeDivisibleConstraint constrains the *body* tile only
        # and lets the border tile be the remainder -- requiring every tile
        # including the remainder to be a multiple over-constrains the solver
        # into picking smaller tiles, which costs more (halo re-fetch) than the
        # alignment saves. Guarded so a dimension smaller than the hardware
        # granularity simply takes the whole dimension instead. Same shape as
        # NE16PWConv2DTileConstraint and the N-EUREKA constraints it came from.
        if parseDict["ch_im_out"] > 32:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "ch_im_out",
                                                      outputChannelVar,
                                                      32,
                                                      strategy = PerformanceHint(priority = 1))
        else:
            tilerModel.addConstraint(outputChannelVar == outputChannelVar.Max(),
                                     strategy = PerformanceHint(priority = 1))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varWeight = operatorRepresentation['weight']
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        replacements: Dict[str, List[int]] = {
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": [],
            "dim_im_in_x_stride": [],
            "dim_im_in_y_stride": [],
            "dim_im_out_x_stride": [],
            "dim_im_out_y_stride": [],
            "input_addr_offset": [],
            "nKo": [],
            "nKi": [],
            "nHo": [],
            "nWo": [],
            "bKo": [],
            "bKi": [],
            "bHo": [],
            "bWo": [],
            "bHi": [],
            "bWi": [],
        }

        replacementTypes = {
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
            "dim_im_in_x_stride": PointerClass(uint32_t),
            "dim_im_in_y_stride": PointerClass(uint32_t),
            "dim_im_out_x_stride": PointerClass(uint32_t),
            "dim_im_out_y_stride": PointerClass(uint32_t),
            "input_addr_offset": PointerClass(uint32_t),
            "nKo": PointerClass(uint16_t),
            "nKi": PointerClass(uint16_t),
            "nHo": PointerClass(uint16_t),
            "nWo": PointerClass(uint16_t),
            "bKo": PointerClass(uint16_t),
            "bKi": PointerClass(uint16_t),
            "bHo": PointerClass(uint16_t),
            "bWo": PointerClass(uint16_t),
            "bHi": PointerClass(uint16_t),
            "bWi": PointerClass(uint16_t),
        }

        weightH = operatorRepresentation['dim_kernel_y']
        weightW = operatorRepresentation['dim_kernel_x']
        weightC = operatorRepresentation['ch_im_in']

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        outputBuffer = ctxt.lookup(varOut)
        assert isinstance(outputBuffer, VariableBuffer)

        for cube in outputCubes:
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            InCube, padding_tuple = Conv2DTileConstraint.computeInputCube((weightH, weightW), pads, strides, weightC,
                                                                          cube,
                                                                          ctxt.lookup(varOut).shape)

            # computeInputCube hard-codes the input channel range to
            # (offset 0, size inputCSize) because dense convolution never tiles
            # its input channels -- they are pinned to the full extent. Depthwise
            # does tile them: each output channel is produced from exactly one
            # input channel, so an output tile covering channels [COffset,
            # COffset + CSize) must read precisely that slice. Left uncorrected,
            # the second channel tile reads from offset 0 -- the wrong channels
            # entirely -- and the first one over-reads past its own tile.
            InCube = HyperRectangle(InCube.offset[:-1] + (COffset,), InCube.dims[:-1] + (CSize,))
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inBSize, inHSize, inWSize, inCSize = InCube.dims

            dim_im_in_x_stride, dim_im_in_y_stride = ioStridesFromDimensions(inWSize, inCSize,
                                                                             operatorRepresentation["input_bits"])
            replacements['dim_im_in_x_stride'].append(dim_im_in_x_stride)
            replacements['dim_im_in_y_stride'].append(dim_im_in_y_stride)
            dim_im_out_x_stride, dim_im_out_y_stride = ioStridesFromDimensions(WSize, CSize,
                                                                               operatorRepresentation["output_bits"])
            replacements['dim_im_out_x_stride'].append(dim_im_out_x_stride)
            replacements['dim_im_out_y_stride'].append(dim_im_out_y_stride)

            replacements['input_addr_offset'].append(
                getInputAddrOffset(inWSize, dim_im_in_y_stride, padding_top, padding_left))

            nKo, nKi, nHo, nWo, bKo, bKi, bHo, bWo, bHi, bWi = NE162DDWConvTemplate.getCounters(
                inCSize, HSize, WSize, CSize, padding_bottom, padding_right, operatorRepresentation)

            replacements["nKo"].append(nKo)
            replacements["nKi"].append(nKi)
            replacements["nHo"].append(nHo)
            replacements["nWo"].append(nWo)
            replacements["bKo"].append(bKo)
            replacements["bKi"].append(bKi)
            replacements["bHo"].append(bHo)
            replacements["bWo"].append(bWo)
            replacements["bHi"].append(bHi)
            replacements["bWi"].append(bWi)

            inputInCubes.append(InCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a in inputInCubes:
            inputLoadSchedule.append({"data_in": a})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        weightBuffer = ctxt.lookup(varWeight)
        assert isinstance(weightBuffer, VariableBuffer)
        weightShape = weightBuffer.shape

        if hasattr(weightBuffer, "_memoryLevel") and weightBuffer._memoryLevel == "WeightMemory_SRAM":
            replacements['weight_addr_offset'] = []
            replacementTypes['weight_addr_offset'] = PointerClass(uint32_t)
            for _ in absoluteOutputCubes:
                # DW weight is a single packed block — no per-cout offset.
                replacements['weight_addr_offset'].append(0)
        else:
            inputWeightBaseOffsets, outputWeightBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                                  operatorRepresentation, ['weight'])
            inputBaseOffsets.update(inputWeightBaseOffsets)
            outputBaseOffsets.update(outputWeightBaseOffsets)

            # DW weight is a single packed (1, 1, packed_bytes) block used
            # across all output-channel tiles — same cube every iteration.
            for _cube, load in zip(outputCubes, inputLoadSchedule):
                load['weight'] = HyperRectangle((0, 0, 0), (weightShape[0], weightShape[1], weightShape[2]))

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class NE16RQSDWConv2DTileConstraint(NE16DWConv2DTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = NE16DWConv2DTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)
        return requantAddGeometricalConstraint(tilerModel, parseDict, ctxt)

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        variableReplacementSchedule, tilingSchedule = super().serializeTilingSolution(
            tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation)

        addrNames = ['mul', 'add']
        inputRequantBaseOffsets, _ = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation,
                                                         addrNames)
        newInputBaseOffsets = {**tilingSchedule.inputBaseOffsets, **inputRequantBaseOffsets}

        requantSchedule = requantLoadSchedule(absoluteOutputCubes, ctxt, operatorRepresentation)
        newInputLoadSchedule = [{
            **load,
            **rqLoad
        } for load, rqLoad in zip(tilingSchedule.inputLoadSchedule, requantSchedule)]

        newTilingSchedule = TilingSchedule(newInputBaseOffsets, tilingSchedule.outputBaseOffsets, newInputLoadSchedule,
                                           tilingSchedule.outputLoadSchedule)

        return variableReplacementSchedule, newTilingSchedule
