# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class RQDWConv2DTileConstraint(TileConstraint):

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

        # SCHEREMO: NCHW layout

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)

        # SCHEREMO: CHW layout

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)

        # SCHEREMO: NHWC layout

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)
        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        # map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)  # Batch
        tilerModel.addConstraint(outputChannelVar == weightOutChannelVar)  # Output Channel
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)  # Input Channel

        tilerModel.addConstraint(outputChannelVar == addChannelVar)
        tilerModel.addConstraint(outputChannelVar == mulChannelVar)

        inputBuffer = ctxt.lookup(inputBufferName)

        effectiveHeight = inputHeightVar + ((padding[0] + padding[2]) * (inputHeightVar == inputBuffer.shape[2]))
        effectiveWidth = inputWidthVar + ((padding[1] + padding[3]) * (inputWidthVar == inputBuffer.shape[3]))

        tilerModel.addConstraint((outputHeightVar == (effectiveHeight - (weightHeightVar - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint((outputWidthVar == (effectiveWidth - (weightWidthVar - 1) - 1) // strides[1] + 1))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 2)

        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)

        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)

        # SCHEREMO: Work around tiling issue with non-wordaligned accesses
        if "L3" in ctxt.lookup(parseDict['data_in'])._memoryLevel:
            tilerModel.addTileSizeDivisibleConstraint(parseDict, 'ch_im_in', inputChannelVar, 4)

        strides = parseDict["strides"]
        pads = parseDict["pads"]

        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])

        # VIC: Constraint the minimum tile size such that we can apply at least one kernel on it
        # SCHEREMO: Account for padding - needed for MobileNetv1
        tilerModel.addConstraint(outputHeightVar >= 1 + max([pads[0], pads[2]]))
        tilerModel.addConstraint(outputWidthVar >= 1 + max([pads[1], pads[3]]))

        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'] + pads[0], strategy = PerformanceHint(1))
        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_y'] + pads[1], strategy = PerformanceHint(1))

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['ch_im_in'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
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
            "ch_im_in": [],
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
            "ch_im_in": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t)
        }

        weightH = ctxt.lookup(varWeight).shape[1]
        weightW = ctxt.lookup(varWeight).shape[2]

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        for cube in outputCubes:

            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            NHWCInCube, padding_tuple = Conv2DTileConstraint.computeInputCube((weightH, weightW), pads, strides, CSize,
                                                                              cube,
                                                                              ctxt.lookup(varOut).shape)
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            NCHWInCube = HyperRectangle((NHWCInCube.offset[0], COffset, NHWCInCube.offset[1], NHWCInCube.offset[2]),
                                        (NHWCInCube.dims[0], CSize, NHWCInCube.dims[1], NHWCInCube.dims[2]))

            RequantCube = HyperRectangle((COffset,), (CSize,))
            WeightCube = HyperRectangle((COffset, 0, 0, 0), (CSize, weightH, weightW, 1))

            replacements['dim_im_in_x'].append(NCHWInCube.dims[2])
            replacements['dim_im_in_y'].append(NCHWInCube.dims[3])
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)
            replacements['ch_im_in'].append(CSize)

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inputInCubes.append(NCHWInCube)
            inputAddCubes.append(RequantCube)
            inputMulCubes.append(RequantCube)
            inputWeightCubes.append(WeightCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, b, add, mul in zip(inputInCubes, inputWeightCubes, inputAddCubes, inputMulCubes):
            inputLoadSchedule.append({"data_in": a, "weight": b, "add": add, "mul": mul})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class DWConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        '''
        This function adds geometrical constraints for a PULP Im2Col 2D DW Convolution Tilling.
        '''

        # ===== GET NECESSARY INFORMATION =====
        # Get to-be-tiled tensor's buffers
        inputBufferName = parseDict['data_in']        
        outputBufferName = parseDict['data_out']

        weightBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        im2colBufferName = parseDict['ctxtBuffer']

        # Get other information
        has_bias = False if parseDict['has_bias'] == "false" else True

        pads = parseDict['pads']
        strides = parseDict['strides']
        dilations = parseDict['dilations']
        group = parseDict['group']

        n_cores = ctxt.n_cores
        im2col_buffer_size = ctxt.lookup(im2colBufferName).size
        weight_type_width = ctxt.lookup(weightBufferName)._type.typeWidth // 8

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
        #   C_out - C_in - H - W layout (depthwise convolution weights,
        #   with c_in used for grouping different than number of channels)
        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 3)

        #   Bias (C_out)
        if has_bias:
            biasDimVar = tilerModel.getTensorDimVar(tensorName = biasBufferName, dimIdx = 0)

        # ===== ADD CONSTRAINTS =====
        #   Add constraint for batch size match between input and output
        tilerModel.addConstraint(inputBatchVar == outputBatchVar)

        #   Add constraint for input width and height sizes match
        #   (Depends on output height and width, kernel size, padding, dilations, and strides.
        #   For more information on the connections, see ONNX and/or Torch Conv2D documentation).
        tilerModel.addConstraint(outputHeightVar == (((inputHeightVar + pads[0] + pads[2] - dilations[0] * (weightHeightVar - 1) -1) // strides[0]) + 1))
        tilerModel.addConstraint(outputWidthVar == (((inputWidthVar + pads[1] + pads[3] - dilations[1] * (weightWidthVar - 1) -1) // strides[1]) + 1))

        #   Add constraint for input channel size match
        #   (Depends on weight output channel and conv grouping)
        tilerModel.addConstraint(inputChannelVar == (weightInChannelVar * group))

        #   Add constraint for weight output channels to match
        #   output number of channels
        tilerModel.addConstraint(weightOutChannelVar == outputChannelVar)

        #   Add constraint for bias size to match number of output channels
        if has_bias:
            tilerModel.addConstraint(biasDimVar == outputChannelVar)

        #   Add constraint for size of im2col buffer to be equal to
        #   number of cores * width of weight data type * size of a single convolutional filter
        tilerModel.addConstraint(im2col_buffer_size == (n_cores * weight_type_width * weightHeightVar * weightWidthVar))

        #   Add constraint for relationship between in and out number of channels
        tilerModel.addConstraint((outputChannelVar % inputChannelVar) == 0)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # ===== GET NECESSARY INFORMATION =====
        # Get to-be-tiled tensor's buffers
        inputBufferName = parseDict['data_in']        
        outputBufferName = parseDict['data_out']

        weightBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        # Get other information
        has_bias = False if parseDict['has_bias'] == "false" else True

        pads = parseDict['pads']
        strides = parseDict['strides']

        # ===== ADD I/O DIMS TO MODEL AS VARS =====
        buffersOfInterest = [inputBufferName, outputBufferName, weightBufferName]
        if has_bias:
            buffersOfInterest.append(biasBufferName)

        for bufferName in buffersOfInterest:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # ===== EXTRACT TENSOR DIMS AS VARS =====
        #   Input
        #   NHWC layout
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        #   Output
        #   NHWC layout
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)

        #   Weight
        #   C_out - C_in - H - W layout (depthwise convolution weights,
        #   with c_in used for grouping different than number of channels)
        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 3)

        #   Bias (C_out)
        if has_bias:
            biasDimVar = tilerModel.getTensorDimVar(tensorName = biasBufferName, dimIdx = 0)

        # ===== ADD CONSTRAINTS =====
        # Workaround tiling issue with non-wordaligned accesses
        if "L3" in ctxt.lookup(parseDict['data_in'])._memoryLevel:
            tilerModel.addTileSizeDivisibleConstraint(parseDict, 'ch_im_in', inputChannelVar, 4)

        # Check that height and width of weights match the parsed values
        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])
        tilerModel.addConstraint(weightOutChannelVar == parseDict['ch_im_out'])

        # Check bias dimension
        if biasBufferName != "NULL":
            tilerModel.addConstraint(biasDimVar == parseDict["ch_im_out"])

        # Constraint the minimum tile size such that at least one kernel can be applied
        # Account for padding
        tilerModel.addConstraint(outputHeightVar >= 1 + max([pads[0], pads[2]]))
        tilerModel.addConstraint(outputWidthVar >= 1 + max([pads[1], pads[3]]))

        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'] + pads[0], strategy = PerformanceHint(1))
        tilerModel.addConstraint(inputWidthVar >= parseDict['dim_kernel_y'] + pads[1], strategy = PerformanceHint(1))

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['ch_im_in'] = tilerModel.getTensorDimVar(inputBuffer.name, 3)
        symbolicParseDict['dim_kernel_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 3)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        # Extract rectangle information (offsets and dimensions) from output cubes
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Extract required component information from operator representation
        varIn = operatorRepresentation['data_in']
        varWeight = operatorRepresentation['weight']
        varBias = operatorRepresentation['bias']
        varOut = operatorRepresentation['data_out']

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
            "ch_im_out": [],
            "ch_im_in": [],
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
            "ch_im_in": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t)
        }

        # Obtain weight dimensions
        # C_out - C_in - H - W layout (depthwise convolution weights,
        # with c_in used for grouping different than number of channels)
        weightC_in = ctxt.lookup(varWeight).shape[1]
        weightH = ctxt.lookup(varWeight).shape[2]
        weightW = ctxt.lookup(varWeight).shape[3]

        # Obtain padding and striding information
        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']
        group = operatorRepresentation['group']

        # Iterate throught the cubes in which the output will be split for tiling
        for cube in outputCubes:
            # Obtain current cube offsets and dimensions
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            # Compute input cube
            InCube, padding_tuple = Conv2DTileConstraint.computeInputCube(kernelShape = (weightH, weightW),
                                                                          pads = pads,
                                                                          strides = strides,
                                                                          inputCSize = weightC_in * group,
                                                                          outputCube = cube,
                                                                          inputDims = ctxt.lookup(varIn).shape,
                                                                          outputDims = ctxt.lookup(varOut).shape)

            # Extract individual padding
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            # Extract InCuve hyperrectangle
            InCube = HyperRectangle((InCube.offset[0], InCube.offset[1], InCube.offset[2], InCube.offset[3]),
                                    (InCube.dims[0], InCube.dims[1], InCube.dims[2], InCube.dims[3]))

            # Prepare weight cube
            WeightCube = HyperRectangle((COffset, 0, 0, 0), (CSize, InCube.dims[3] // group, weightH, weightW))

            # Add element information for the operator representation
            replacements['dim_im_in_x'].append(InCube.dims[1])
            replacements['dim_im_in_y'].append(InCube.dims[2])
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)
            replacements['ch_im_in'].append(InCube.dims[3])

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            # Add computed cubes to the respective lists
            inputInCubes.append(InCube)
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
