# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class RQConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        '''
        This function add geometrical constraints for a PULP Im2Col Convolution Tilling.
        '''

        # Get to-be-tiled tensor's buffers
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']
        outputBufferName = parseDict['data_out']

        strides = parseDict["strides"]
        padding = parseDict["pads"]
        dilation = parseDict["dilations"]

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, weightBufferName, mulBufferName, addBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 3)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)
        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)  # Batch
        tilerModel.addConstraint(outputChannelVar == weightOutChannelVar)  # Output Channel

        tilerModel.addConstraint(outputChannelVar == addChannelVar)
        tilerModel.addConstraint(outputChannelVar == mulChannelVar)

        inputBuffer = ctxt.lookup(inputBufferName)

        effectiveHeight = inputHeightVar + ((padding[0] + padding[2]) * (inputHeightVar == inputBuffer.shape[1]))
        effectiveWidth = inputWidthVar + ((padding[1] + padding[3]) * (inputWidthVar == inputBuffer.shape[2]))

        tilerModel.addConstraint((outputHeightVar == (effectiveHeight - (weightHeightVar - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint((outputWidthVar == (effectiveWidth - (weightWidthVar - 1) - 1) // strides[1] + 1))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        outputChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 3)

        strides = parseDict["strides"]
        padding = parseDict["pads"]

        # VIC: Force at least one row of A and one col of B in the GEMM (since it's a im2col Conv) to avoid partial results
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])

        if (parseDict["ch_im_out"] >= 8):
            tilerModel.addMinTileSizeConstraint(parseDict, 'ch_im_out', outputChannelVar, 8)

        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(inputWidthVar >= parseDict['dim_kernel_y'])
        tilerModel.addConstraint(weightInChannelVar == parseDict['ch_im_in'])

        # VIC: Constraint the minimum tile size such that we can apply at least one kernel on it
        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(inputWidthVar >= parseDict['dim_kernel_y'])

        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['dim_im_in_x'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        symbolicParseDict['dim_kernel_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weight', 'mul', 'add', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varWeight = operatorRepresentation['weight']
        varIn = operatorRepresentation["data_in"]
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        inputAddCubes = []
        inputMulCubes = []
        inputWeightCubes = []
        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": []
        }

        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t)
        }

        weightH = ctxt.lookup(varWeight).shape[1]
        weightW = ctxt.lookup(varWeight).shape[2]
        weightC = ctxt.lookup(varWeight).shape[3]

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        for cube in outputCubes:
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            InCube, padding_tuple = Conv2DTileConstraint.computeInputCube(
                kernelShape = (weightH, weightW),
                pads = pads,
                strides = strides,
                inputCSize = weightC,
                outputCube = cube,
                inputDims = ctxt.lookup(varIn).shape,
                outputDims = ctxt.lookup(varOut).shape,
            )

            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            replacements['dim_im_in_x'].append(InCube.dims[1])
            replacements['dim_im_in_y'].append(InCube.dims[2])
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inputInCubes.append(InCube)

            RequantCube = HyperRectangle((COffset,), (CSize,))
            WeightCube = HyperRectangle((COffset, 0, 0, 0), (CSize, weightH, weightW, weightC))

            inputWeightCubes.append(WeightCube)
            inputAddCubes.append(RequantCube)
            inputMulCubes.append(RequantCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, b, add, mul in zip(inputInCubes, inputWeightCubes, inputAddCubes, inputMulCubes):
            inputLoadSchedule.append({"data_in": a, "weight": b, "add": add, "mul": mul})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class Conv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Add geometrical constraints for Conv2D tiling.

        For spatial tiling, input tiles require extra memory for overlap regions
        at tile boundaries (kernel receptive field). This method accounts for worst-case
        overlap on all sides.

        Future optimization: Currently uses worst-case memory allocation (kernel_size - 1
        on all sides). A more memory-efficient approach would compute exact
        per-tile memory requirements during serializeTilingSolution based on actual tile
        positions, but this requires more extensive framework changes.
        """

        # ===== GET NECESSARY INFORMATION =====
        #   Get to-be-tiled tensor buffers
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        weightBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        inputBuffer = ctxt.lookup(inputBufferName)

        #   Get other information
        has_bias = False if parseDict['has_bias'] == "false" else True

        pads = parseDict["pads"]
        strides = parseDict["strides"]
        dilations = parseDict["dilations"]
        group = parseDict["group"]

        # ===== ADD I/O DIMS TO MODEL AS VARS =====
        buffersOfInterest = [inputBufferName, outputBufferName, weightBufferName]
        if has_bias:
            buffersOfInterest.append(biasBufferName)

        for bufferName in buffersOfInterest:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # ===== EXTRACT TENSOR DIMS AS VARS =====
        #   Input
        #   NHWC layout
        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        #   Output
        #   NHWC layout
        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        #   Weight
        #   C_out - H - W layout - C_in
        #   (with c_in used for grouping different than number of channels)
        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 3)

        #   Bias (C_out)
        if has_bias:
            biasDimVar = tilerModel.getTensorDimVar(tensorName = biasBufferName, dimIdx = 0)

        # ===== COMPUTE EFFECTIVE INPUT HEIGHT AND WIDTH =====
        #   Assume worst case scenario (data padding on all sides) when tiling on a ceratin dimension.
        effectiveInputHeight = inputHeightVar + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[1])) - (
            (weightHeightVar - 1) * (inputHeightVar != inputBuffer.shape[1]))
        effectiveInputWidth = inputWidthVar + ((pads[1] + pads[3]) * (inputWidthVar == inputBuffer.shape[2])) - (
            (weightWidthVar - 1) * (inputWidthVar != inputBuffer.shape[2]))

        # ===== ADD CONSTRAINTS =====
        #   Add constraint for batch size match between input and output
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)

        #   Add constraint for input width and height sizes match
        #   (Depends on output height and width, kernel size, padding, dilations, and strides.
        #   For more information on the connections, see ONNX and/or Torch Conv2D documentation).
        tilerModel.addConstraint(
            (outputHeightVar == (effectiveInputHeight - dilations[0] * (weightHeightVar - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint(
            (outputWidthVar == (effectiveInputWidth - dilations[1] * (weightWidthVar - 1) - 1) // strides[1] + 1))

        #   Add constraint for input channel size match
        #   (Depends on weight output channel and conv grouping)
        tilerModel.addConstraint(inputChannelVar == (weightInChannelVar * group))

        #   Add constraint for weight output channels to match
        #   output number of channels
        tilerModel.addConstraint(weightOutChannelVar == outputChannelVar)

        #   Add constraint for bias size to match number of output channels
        if has_bias:
            tilerModel.addConstraint(biasDimVar == outputChannelVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # ===== GET NECESSARY INFORMATION =====
        #   Get to-be-tiled tensor buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        #   Get other information
        pads = parseDict["pads"]
        strides = parseDict["strides"]

        # ===== EXTRACT TENSOR DIMS AS VARS =====
        #   Input
        #   NHWC layout
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        #   Weight
        #   C_out - H - W layout - C_in
        #   (with c_in used for grouping different than number of channels)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 3)

        # ===== COMPUTE EFFECTIVE INPUT HEIGHT AND WIDTH =====
        #   Assume worst case scenario (data padding on all sides) when tiling on a ceratin dimension.
        effectiveInputHeight = inputHeightVar + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[1])) - (
            (weightHeightVar - 1) * (inputHeightVar != inputBuffer.shape[1]))
        effectiveInputWidth = inputWidthVar + ((pads[1] + pads[3]) * (inputWidthVar == inputBuffer.shape[2])) - (
            (weightWidthVar - 1) * (inputWidthVar != inputBuffer.shape[2]))

        # ===== ADD CONSTRAINTS =====
        #   Keep whole input channels (required for im2col algorithm)
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])

        #   Require minimum input spatial dimensions to be at least kernel size for proper convolution application
        tilerModel.addConstraint(effectiveInputHeight >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(effectiveInputWidth >= parseDict['dim_kernel_y'])

        #   Ensure input tiles are compatible with stride
        tilerModel.addConstraint((effectiveInputHeight % strides[0]) == 0)
        tilerModel.addConstraint((effectiveInputWidth % strides[1]) == 0)

        #   Weight should not be tiled
        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])
        tilerModel.addConstraint(weightInChannelVar * parseDict['group'] == parseDict['ch_im_in'])

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        symbolicParseDict = parseDict.copy()

        symbolicParseDict['dim_im_in_x'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        symbolicParseDict['dim_im_in_y'] = tilerModel.getTensorDimVar(inputBuffer.name, 2)

        symbolicParseDict['dim_kernel_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)

        symbolicParseDict['dim_im_out_x'] = tilerModel.getTensorDimVar(outputBuffer.name, 1)
        symbolicParseDict['dim_im_out_y'] = tilerModel.getTensorDimVar(outputBuffer.name, 2)

        return symbolicParseDict

    @staticmethod
    def computeInputCube(
        kernelShape: Tuple[int, int],
        pads: Tuple[int, int, int, int],
        strides: Tuple[int, int],
        inputCSize: int,
        outputCube: HyperRectangle,
        outputDims: Tuple[int, int, int],
        inputDims: Optional[Tuple[int, int, int]] = None,
        outputAbsoluteOffsets: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:

        # Obtain relative and absolute information about the output tile
        (outputBatchOffset, outputHOffset, outputWOffset, _) = outputCube.offset
        (outputBatchSize, outputHSize, outputWSize, _) = outputCube.dims
        (_, outputHAbsoluteOffset, outputWAbsoluteOffset,
         _) = outputAbsoluteOffsets if outputAbsoluteOffsets is not None else outputCube.offset

        # Extract individual pads and strides
        padTop, padLeft, padBottom, padRight = pads
        strideH, strideW = strides

        # Compute actuale tile padding, depending on tile position (keep padding only for margins situated at the edge).
        # Required for the Im2Col kernel that handles 0-padding internally.
        tilePadTop = padTop if (outputHAbsoluteOffset == 0) else 0
        tilePadLeft = padLeft if (outputWAbsoluteOffset == 0) else 0
        tilePadBottom = padBottom if (outputHAbsoluteOffset + outputHSize == outputDims[1]) else 0
        tilePadRight = padRight if (outputWAbsoluteOffset + outputWSize == outputDims[2]) else 0

        # LMACAN: Calculating the per-dimension relative tile offset without padding
        #         The offset is relative to the upstream bigger tile, and represents the offset to
        #         "useful" data, so padding is not included.
        inputHOffset = max(outputHOffset * strideH - padTop, 0)
        inputWOffset = max(outputWOffset * strideW - padLeft, 0)

        # Compute input dimensions according to procedure described in PyTorch's Conv2D documentation
        # Assuming worst case (cutting of (stride - 1) elements at the end of each dimension)
        inputHSize = outputHSize * strideH + (kernelShape[0] - 1) - (tilePadTop + tilePadBottom)
        inputWSize = outputWSize * strideW + (kernelShape[1] - 1) - (tilePadLeft + tilePadRight)

        if inputDims is not None:
            # Clamp to remaining input size from the current offset
            # This prevents reading beyond input boundaries for edge tiles
            inputHSize = min(inputHSize, inputDims[1] - inputHOffset)
            inputWSize = min(inputWSize, inputDims[2] - inputWOffset)

        # Generate input tile object
        InCube = HyperRectangle((outputBatchOffset, inputHOffset, inputWOffset, 0),
                                (outputBatchSize, inputHSize, inputWSize, inputCSize))

        return InCube, (tilePadLeft, tilePadRight, tilePadTop, tilePadBottom)

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        # Extract rectangle information (offsets and dimensions) from output cubes
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Extract required component information from operator representation
        varIn = operatorRepresentation["data_in"]
        varWeight = operatorRepresentation['weight']
        varBias = operatorRepresentation['bias']
        varOut = operatorRepresentation['data_out']

        group = operatorRepresentation["group"]

        # Prepare address names, also handling bias
        if varBias != "NULL":
            addrNames = ['data_in', 'weight', 'bias', 'data_out']
        else:
            addrNames = ['data_in', 'weight', 'data_out']

        # Extract memory base addresses for each of the required components,
        # based on the computed memory configuration
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # Prepare cube lists for components
        inputInCubes = []
        inputWeightCubes = []
        inputBiasCubes = []

        # Prepare replacement lists for the elements inside the operator representation,
        # for the cubes to be computed further down in this function
        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": []
        }

        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t)
        }

        # Obtain weight dimensions
        (_, weightH, weightW, weightCin) = ctxt.lookup(varWeight).shape

        # Obtain padding and striding information
        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        # Iterate throught the cubes in which the output will be split for tiling
        for idx, cube in enumerate(outputCubes):
            # Obtain current cube offsets and dimensions
            COffset = cube.offset[3]
            (_, HSize, WSize, CSize) = cube.dims

            # Compute input cube
            InCube, padding_tuple = Conv2DTileConstraint.computeInputCube(
                kernelShape = (weightH, weightW),
                pads = pads,
                strides = strides,
                inputCSize = weightCin * group,
                outputCube = cube,
                inputDims = ctxt.lookup(varIn).shape,
                outputDims = ctxt.lookup(varOut).shape,
                outputAbsoluteOffsets = absoluteOutputCubes[idx].absoluteOffset)

            # Extract individual padding
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            # Add element information for the operator representation
            replacements['dim_im_in_x'].append(InCube.dims[1])
            replacements['dim_im_in_y'].append(InCube.dims[2])

            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)

            replacements['ch_im_in'].append(weightCin * group)
            replacements['ch_im_out'].append(CSize)

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            # Add input cube with tiling information to the corresponding list
            inputInCubes.append(InCube)

            # Obtain and add weight cube with tiling information to the corresponding list
            WeightCube = HyperRectangle((COffset, 0, 0, 0), (CSize, weightH, weightW, weightCin))
            inputWeightCubes.append(WeightCube)

            # Obtain and add bias cube with tiling information to the corresponding list,
            # if bias exists
            if varBias != "NULL":
                BiasCube = HyperRectangle((COffset,), (CSize,))
                inputBiasCubes.append(BiasCube)

        # Prepare loading schedule lists
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Create input schedule lists, with bias handling
        if varBias == "NULL":
            for a, b in zip(inputInCubes, inputWeightCubes):
                inputLoadSchedule.append({"data_in": a, "weight": b})
        else:
            for a, b, c in zip(inputInCubes, inputWeightCubes, inputBiasCubes):
                inputLoadSchedule.append({"data_in": a, "weight": b, "bias": c})

        # Create output schedule list
        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        # Prepare containing objects with information computed in this function regarding tiling schedule
        # and variable replacement inside operator representation
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class RQConv1DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        '''
        Add geometrical constraints for a PULP Im2Col Convolution Tiling.
        '''

        # Get to-be-tiled tensor's buffers
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']
        outputBufferName = parseDict['data_out']

        pads = parseDict["pads"]
        stride = parseDict["strides"][0]
        dilation = parseDict["dilations"][0]

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, weightBufferName, mulBufferName, addBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputLengthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightLengthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputLengthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)

        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)
        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)  # Batch
        tilerModel.addConstraint(outputChannelVar == weightOutChannelVar)  # Output Channel
        tilerModel.addConstraint(outputChannelVar == addChannelVar)
        tilerModel.addConstraint(outputChannelVar == mulChannelVar)

        inputBuffer = ctxt.lookup(inputBufferName)
        effectiveLength = inputLengthVar + (sum(pads) * (inputLengthVar == inputBuffer.shape[1]))
        tilerModel.addConstraint((outputLengthVar == (effectiveLength - (weightLengthVar - 1) - 1) // stride + 1))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        pads = parseDict["pads"]
        stride = parseDict["strides"][0]

        inputLengthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)

        outputChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)
        weightLengthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)

        # VIC: Force at least one row of A and one col of B in GEMM (since it's a im2col Conv) to avoid partial results
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])

        if (parseDict["ch_im_out"] >= 8):
            tilerModel.addMinTileSizeConstraint(parseDict, 'ch_im_out', outputChannelVar, 8)

        tilerModel.addConstraint(inputLengthVar >= parseDict['dim_kernel_y'])
        tilerModel.addConstraint(weightInChannelVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(weightLengthVar == parseDict['dim_kernel_y'])
        tilerModel.addConstraint((inputLengthVar % stride) == 0)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['dim_im_in_y'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        symbolicParseDict['dim_im_out_y'] = tilerModel.getTensorDimVar(outputBuffer.name, 1)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weight', 'mul', 'add', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varWeight = operatorRepresentation['weight']
        varIn = operatorRepresentation["data_in"]
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        inputAddCubes = []
        inputMulCubes = []
        inputWeightCubes = []
        replacements: Dict[str, List[int]] = {
            "dim_im_in_y": [],
            "dim_im_out_y": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
        }

        replacementTypes = {
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
        }

        weightL = ctxt.lookup(varWeight).shape[1]
        weightC = ctxt.lookup(varWeight).shape[2]

        pads = operatorRepresentation['pads']
        stride = operatorRepresentation['strides'][0]

        for cube in outputCubes:
            (_, _, COffset) = cube.offset
            (_, LSize, CSize) = cube.dims

            InCube, (pad_top, pad_bottom) = Conv1DTileConstraint.computeInputCube(
                kernelLength = weightL,
                pads = pads,
                stride = stride,
                inputCSize = weightC,
                outputCube = cube,
                inputDims = ctxt.lookup(varIn).shape,
                outputDims = ctxt.lookup(varOut).shape,
            )

            replacements['dim_im_in_y'].append(InCube.dims[1])
            replacements['dim_im_out_y'].append(LSize)
            replacements['ch_im_out'].append(CSize)
            replacements['padding_y_top'].append(pad_top)
            replacements['padding_y_bottom'].append(pad_bottom)

            inputInCubes.append(InCube)

            RequantCube = HyperRectangle((COffset,), (CSize,))
            WeightCube = HyperRectangle((COffset, 0, 0), (CSize, weightL, weightC))

            inputWeightCubes.append(WeightCube)
            inputAddCubes.append(RequantCube)
            inputMulCubes.append(RequantCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, b, add, mul in zip(inputInCubes, inputWeightCubes, inputAddCubes, inputMulCubes):
            inputLoadSchedule.append({"data_in": a, "weight": b, "add": add, "mul": mul})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class Conv1DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Add geometrical constraints for Conv1D tiling.
        """

        # ===== GET NECESSARY INFORMATION =====
        #   Get to-be-tiled tensor buffers
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        weightBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        inputBuffer = ctxt.lookup(inputBufferName)

        #   Get other information
        has_bias = False if parseDict['has_bias'] == "false" else True

        pads = parseDict["pads"]
        stride = parseDict["strides"][0]
        dilation = parseDict["dilations"][0]
        group = parseDict["group"]

        # ===== ADD I/O DIMS TO MODEL AS VARS =====
        buffersOfInterest = [inputBufferName, outputBufferName, weightBufferName]
        if has_bias:
            buffersOfInterest.append(biasBufferName)

        for bufferName in buffersOfInterest:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # ===== EXTRACT TENSOR DIMS AS VARS =====
        #   Input
        #   NLC layout
        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputLengthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)

        #   Output
        #   NLC layout
        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputLengthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)

        #   Weight
        #   C_out - L - C_in
        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightLengthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)

        #   Bias (C_out)
        if has_bias:
            biasDimVar = tilerModel.getTensorDimVar(tensorName = biasBufferName, dimIdx = 0)

        # ===== ADD CONSTRAINTS =====

        #   Assume worst case scenario (data padding on all sides) when tiling on a ceratin dimension.
        effectiveInputLengthVar = inputLengthVar \
            + (sum(pads) * (inputLengthVar == inputBuffer.shape[1])) \
            - ((inputLengthVar - 1) * (inputLengthVar != inputBuffer.shape[1]))

        _outputLengthVar = (effectiveInputLengthVar - dilation * (weightLengthVar - 1) - 1) // stride + 1

        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(inputChannelVar == (weightInChannelVar * group))
        tilerModel.addConstraint(weightOutChannelVar == outputChannelVar)
        tilerModel.addConstraint(outputLengthVar == _outputLengthVar)
        if has_bias:
            tilerModel.addConstraint(biasDimVar == outputChannelVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        pads = parseDict["pads"]
        stride = parseDict["strides"][0]
        group = parseDict['group']

        inputLengthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)

        weightLengthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)

        effectiveInputLength = inputLengthVar \
            + (sum(pads) * (inputLengthVar == inputBuffer.shape[1])) \
            - (inputLengthVar - 1) * (inputLengthVar != inputBuffer.shape[1])

        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(effectiveInputLength >= parseDict['dim_kernel_y'])
        tilerModel.addConstraint((effectiveInputLength % stride) == 0)
        tilerModel.addConstraint(weightLengthVar == parseDict['dim_kernel_y'])
        tilerModel.addConstraint(weightInChannelVar * group == parseDict['ch_im_in'])

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['dim_im_in_y'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        symbolicParseDict['dim_im_out_y'] = tilerModel.getTensorDimVar(outputBuffer.name, 1)

        return symbolicParseDict

    @staticmethod
    def computeInputCube(
        kernelLength: int,
        pads: Tuple[int, int],
        stride: int,
        inputCSize: int,
        outputCube: HyperRectangle,
        outputDims: Tuple[int, int],
        inputDims: Optional[Tuple[int, int]] = None,
        outputAbsoluteOffsets: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[HyperRectangle, Tuple[int, int]]:

        if outputAbsoluteOffsets is None:
            outputAbsoluteOffsets = outputCube.offset

        # Obtain relative and absolute information about the output tile
        (outputBatchOffset, outputLOffset, _) = outputCube.offset
        (outputBatchSize, outputLSize, _) = outputCube.dims
        (_, outputLAbsoluteOffset, _) = outputAbsoluteOffsets

        # Extract individual pads and strides
        padTop, padBottom = pads

        # Compute actual tile padding, depending on tile position (keep padding only for margins situated at the edge).
        # Required for the Im2Col kernel that handles 0-padding internally.
        tilePadTop = padTop if (outputLAbsoluteOffset == 0) else 0
        tilePadBottom = padBottom if (outputLAbsoluteOffset + outputLSize == outputDims[1]) else 0

        # LMACAN: Calculating the per-dimension relative tile offset without padding
        #         The offset is relative to the upstream bigger tile, and represents the offset to
        #         "useful" data, so padding is not included.
        inputLOffset = max(outputLOffset * stride - padTop, 0)

        # Compute input dimensions according to procedure described in PyTorch's Conv2D documentation
        # Assuming worst case (cutting of (stride - 1) elements at the end of each dimension)
        inputLSize = outputLSize * stride + (kernelLength - 1) - (tilePadTop + tilePadBottom)

        if inputDims is not None:
            # Clamp to remaining input size from the current offset
            # This prevents reading beyond input boundaries for edge tiles
            inputLSize = min(inputLSize, inputDims[1] - inputLOffset)

        # Generate input tile object
        InCube = HyperRectangle((outputBatchOffset, inputLOffset, 0), (outputBatchSize, inputLSize, inputCSize))

        return InCube, (tilePadTop, tilePadBottom)

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        # Extract rectangle information (offsets and dimensions) from output cubes
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Extract required component information from operator representation
        varIn = operatorRepresentation["data_in"]
        varWeight = operatorRepresentation['weight']
        varBias = operatorRepresentation['bias']
        varOut = operatorRepresentation['data_out']
        group = operatorRepresentation["group"]

        # Prepare address names, also handling bias
        if varBias != "NULL":
            addrNames = ['data_in', 'weight', 'bias', 'data_out']
        else:
            addrNames = ['data_in', 'weight', 'data_out']

        # Extract memory base addresses for each of the required components,
        # based on the computed memory configuration
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # Prepare cube lists for components
        inputInCubes = []
        inputWeightCubes = []
        inputBiasCubes = []

        # Prepare replacement lists for the elements inside the operator representation,
        # for the cubes to be computed further down in this function
        replacements: Dict[str, List[int]] = {
            "dim_im_in_y": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
        }

        replacementTypes = {
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
        }

        # Obtain weight dimensions
        (_, weightL, weightCin) = ctxt.lookup(varWeight).shape

        # Obtain padding and striding information
        pads = operatorRepresentation['pads']
        stride = operatorRepresentation['strides'][0]

        # Iterate throught the cubes in which the output will be split for tiling
        for idx, cube in enumerate(outputCubes):
            # Obtain current cube offsets and dimensions
            (_, _, COffset) = cube.offset
            (_, LSize, CSize) = cube.dims

            # Compute input cube
            InCube, (pad_top, pad_bottom) = Conv1DTileConstraint.computeInputCube(
                kernelLength = weightL,
                pads = pads,
                stride = stride,
                inputCSize = weightCin * group,
                outputCube = cube,
                inputDims = ctxt.lookup(varIn).shape,
                outputDims = ctxt.lookup(varOut).shape,
                outputAbsoluteOffsets = absoluteOutputCubes[idx].absoluteOffset)

            # Add element information for the operator representation
            replacements['dim_im_in_y'].append(InCube.dims[1])
            replacements['dim_im_out_y'].append(LSize)
            replacements['ch_im_in'].append(weightCin * group)
            replacements['ch_im_out'].append(CSize)
            replacements['padding_y_top'].append(pad_top)
            replacements['padding_y_bottom'].append(pad_bottom)

            # Add input cube with tiling information to the corresponding list
            inputInCubes.append(InCube)

            # Obtain and add weight cube with tiling information to the corresponding list
            WeightCube = HyperRectangle((COffset, 0, 0), (CSize, weightL, weightCin))
            inputWeightCubes.append(WeightCube)

            # Obtain and add bias cube with tiling information to the corresponding list,
            # if bias exists
            if varBias != "NULL":
                BiasCube = HyperRectangle((COffset,), (CSize,))
                inputBiasCubes.append(BiasCube)

        # Prepare loading schedule lists
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Create input schedule lists, with bias handling
        if varBias == "NULL":
            for a, b in zip(inputInCubes, inputWeightCubes):
                inputLoadSchedule.append({"data_in": a, "weight": b})
        else:
            for a, b, c in zip(inputInCubes, inputWeightCubes, inputBiasCubes):
                inputLoadSchedule.append({"data_in": a, "weight": b, "bias": c})

        # Create output schedule list
        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        # Prepare containing objects with information computed in this function regarding tiling schedule
        # and variable replacement inside operator representation
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule