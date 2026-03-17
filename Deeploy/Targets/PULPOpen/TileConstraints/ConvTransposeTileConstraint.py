# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class ConvTranspose2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Register all tensors that participate in the tiled operator. The tiler will
        # create symbolic variables for each tensor dimension and solve over them.
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        outputBufferName = parseDict['data_out']

        for bufferName in [inputBufferName, weightBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # Symbolic dimensions for NCHW input and output, and [Cin, Cout/group, Kh, Kw] weights.
        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 3)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        # Geometrical constraints: batch is preserved, the weight Cin must match input Cin,
        # and in this first implementation H/W are not tiled, so they stay equal to the
        # full tensor shapes known from the context.
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(weightInChannelVar == inputChannelVar)
        tilerModel.addConstraint(outputHeightVar == ctxt.lookup(outputBufferName).shape[2])
        tilerModel.addConstraint(outputWidthVar == ctxt.lookup(outputBufferName).shape[3])
        tilerModel.addConstraint(inputHeightVar == ctxt.lookup(inputBufferName).shape[2])
        tilerModel.addConstraint(inputWidthVar == ctxt.lookup(inputBufferName).shape[3])
        tilerModel.addConstraint(weightHeightVar == ctxt.lookup(weightBufferName).shape[2])
        tilerModel.addConstraint(weightWidthVar == ctxt.lookup(weightBufferName).shape[3])

        if parseDict['group'] == 1:
            # For the regular case, the only tiled dimension is Cout, so the output tile
            # channel count must match the weight tile channel count.
            tilerModel.addConstraint(outputChannelVar == weightOutChannelVar)
        else:
            # Grouped ConvTranspose is kept conservative for now: no real Cout slicing is
            # enforced on the weight tensor because the grouped layout needs dedicated handling.
            tilerModel.addConstraint(outputChannelVar == ctxt.lookup(outputBufferName).shape[1])
            tilerModel.addConstraint(weightOutChannelVar == ctxt.lookup(weightBufferName).shape[1])

        if parseDict.get('has_bias', "false") == "true":
            # Bias follows the tiled output channels one-to-one.
            biasBufferName = parseDict['bias']
            tilerModel.addTensorDimToModel(ctxt, biasBufferName)
            biasChannelVar = tilerModel.getTensorDimVar(tensorName = biasBufferName, dimIdx = 0)
            tilerModel.addConstraint(biasChannelVar == outputChannelVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        inputShape = ctxt.lookup(inputBufferName).shape
        outputShape = ctxt.lookup(outputBufferName).shape

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)

        # Policy constraints: keep the full input resident and tile only along output
        # channels. This is the simplest correct strategy for ConvTranspose because it
        # avoids the inverse spatial dependency problem on H/W tiles.
        tilerModel.addConstraint(inputBatchVar == inputShape[0])
        tilerModel.addConstraint(inputChannelVar == inputShape[1])
        tilerModel.addConstraint(outputBatchVar == outputShape[0])

        if parseDict['group'] == 1 and parseDict["ch_im_out"] >= 8:
            # Do not create tiny channel tiles unless necessary. This reduces overhead
            # from DMA/codegen and keeps the kernel granularity reasonable.
            tilerModel.addMinTileSizeConstraint(parseDict, 'ch_im_out', outputChannelVar, 8)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        # The solver gives us output tiles. For each output tile we must reconstruct the
        # exact input/weight/bias tiles that must be transferred into the target memory.
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weight', 'data_out']
        has_bias = operatorRepresentation.get('has_bias', "false") == "true"
        if has_bias:
            addrNames.append('bias')

        # Resolve base addresses in the target memory level (typically L1) for all tensors
        # used by this tiled operator.
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        inputShape = ctxt.lookup(operatorRepresentation['data_in']).shape
        weightShape = ctxt.lookup(operatorRepresentation['weight']).shape

        inputInCubes = []
        inputWeightCubes = []
        inputBiasCubes = []

        replacements = {
            "ch_im_out": [],
            "batch": [],
        }
        replacementTypes = {
            "ch_im_out": PointerClass(uint16_t),
            "batch": PointerClass(uint8_t),
        }

        for cube in outputCubes:
            # Output tiles are NCHW. Since we only tile along C_out, only the channel
            # offset/size are relevant here.
            _, cOffset, _, _ = cube.offset
            nSize, cSize, _, _ = cube.dims

            # Input is kept whole for every tile in this first implementation.
            inputInCubes.append(HyperRectangle((0, 0, 0, 0), tuple(inputShape)))

            if operatorRepresentation['group'] == 1:
                # Slice the weight tensor along the output-channel axis so the kernel sees
                # only the filters needed for the current output tile.
                weightOffset = (0, cOffset, 0, 0)
                weightDims = (weightShape[0], cSize, weightShape[2], weightShape[3])
            else:
                # Conservative grouped fallback: use full weights until grouped slicing is
                # implemented explicitly for ConvTranspose tiling.
                weightOffset = (0, 0, 0, 0)
                weightDims = tuple(weightShape)

            inputWeightCubes.append(HyperRectangle(weightOffset, weightDims))

            if has_bias:
                if operatorRepresentation['group'] == 1:
                    # Bias is sliced exactly like the output channels.
                    inputBiasCubes.append(HyperRectangle((cOffset,), (cSize,)))
                else:
                    inputBiasCubes.append(HyperRectangle((0,), (ctxt.lookup(operatorRepresentation['bias']).shape[0],)))

            # These replacements are injected into the template so each kernel invocation
            # uses the tile-local shape instead of the global layer shape.
            replacements["ch_im_out"].append(cSize)
            replacements["batch"].append(nSize)

        inputLoadSchedule = []
        outputLoadSchedule = []

        # Build the per-tile DMA/load schedule: which tensor cubes must be present before
        # computing each output cube.
        if has_bias:
            for inCube, weightCube, biasCube in zip(inputInCubes, inputWeightCubes, inputBiasCubes):
                inputLoadSchedule.append({"data_in": inCube, "weight": weightCube, "bias": biasCube})
        else:
            for inCube, weightCube in zip(inputInCubes, inputWeightCubes):
                inputLoadSchedule.append({"data_in": inCube, "weight": weightCube})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        return VariableReplacementScheme(replacements, replacementTypes), schedule
