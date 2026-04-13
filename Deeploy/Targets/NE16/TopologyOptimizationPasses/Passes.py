# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from functools import partial
from typing import Generator, List, Tuple

import numpy as np
import numpy.typing as npt
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, SequentialPass, \
    contextagnostic
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RemoveGlobalOutputReshapePass, _createReshape
from Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses.EngineColoringPasses import \
    EngineDiscolorationPass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ReshapeConstOptPass, ReshapeMergePass


def _weightEncode(weight: npt.NDArray[np.uint8], bits: int, depthwise: bool = False) -> npt.NDArray[np.uint8]:
    """NE16 weight encoder, ported from pulp-nnx/test/Ne16Weight.py.

    Expected weight shape: (cout, cin, H, W).
    Output layout: (cout, cinMajor, Bits, H*W, cinMinorBytes) where
    CIN_SUBTILE = 16 (single mode, no 1x1 vs 3x3 split like Neureka).
    """
    _NE16_CIN_SUBTILE = 16

    if depthwise:
        weight = weight.transpose(1, 0, 2, 3)  # Swap cout and cin

    cout, cin, height, width = weight.shape

    # Pad cin to be divisible with CIN_SUBTILE
    if cin % _NE16_CIN_SUBTILE != 0:
        cinPad = _NE16_CIN_SUBTILE - cin % _NE16_CIN_SUBTILE
        weight = np.pad(
            weight,
            ((0, 0), (0, cinPad), (0, 0), (0, 0)),
            "constant",
            constant_values = 0,
        )
        cin = cin + cinPad

    cinMajor = cin // _NE16_CIN_SUBTILE
    cinMinor = _NE16_CIN_SUBTILE

    # (cout, cinMajor, cinMinor, H*W, 1)
    weight = weight.reshape(cout, cinMajor, cinMinor, height * width, 1)
    # (cout, cinMajor, cinMinor, H*W, Bits)
    weight = np.unpackbits(weight, axis = -1, count = bits, bitorder = "little")
    # (cout, cinMajor, Bits, H*W, cinMinor)
    weight = weight.transpose(0, 1, 4, 3, 2)
    # Pack cinMinor bits into bytes — 16 bits = 2 bytes
    weight = weight.reshape(-1, 8)
    weight = np.packbits(weight, axis = -1, bitorder = "little")
    cinMinorBytes = cinMinor // 8
    # Layout rank varies by conv kind:
    #   - Dense 3x3 (!depthwise, kernel 3x3): rank 4
    #       (cout, cinMajor, Bits, H*W*cinMinorBytes)
    #     — NE16DenseConstraint tiles over weight.shape[3].
    #   - PW 1x1 and DW 3x3: rank 3
    #       (cout, cinMajor, Bits*H*W*cinMinorBytes)
    #     — NE16{Pointwise,Depthwise}Constraint don't need a bits dim.
    if not depthwise and height == 3 and width == 3:
        return weight.reshape(cout, cinMajor, bits, height * width * cinMinorBytes)
    return weight.reshape(cout, cinMajor, bits * height * width * cinMinorBytes)


def _ne16_adjust_weight_memory_layout_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                          ne16EngineName: str):
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    if not ("engine" in node.attrs and node.attrs["engine"] == ne16EngineName):
        return graph

    weightTensor = node.inputs[1]

    if not isinstance(weightTensor, gs.Constant):
        return graph

    # Adjust N-EUREKA's weights
    values = weightTensor.values

    # Extract weight offset and translate weights by the offset
    weight_offset = values.min()
    values = values - weight_offset
    node.attrs["weight_offset"] = weight_offset

    if "channels_first" in node.attrs:
        channels_first = node.attrs["channels_first"]
    else:
        channels_first = default_channels_first

    # Weight encode expects channels-first (cout, cin_per_group, H, W)
    if not channels_first:
        values = values.transpose(0, 3, 1, 2)

    bits = 8  # Support only 8 bit weights for now
    if node.attrs['group'] == 1:
        weightTensor.values = _weightEncode(values.astype(np.uint8), bits, depthwise = False)
    else:
        # Depthwise: Deeploy's NHWC pass leaves weight as
        # (cin_per_group=1, cout=group, H, W) after the transpose above;
        # Ne16Weight.py's encode expects standard (cout, cin_per_group, H, W)
        # — swap axes 0/1 before encoding so the result is a single packed
        # (1, 1, packed_bytes) block across up to NE16_SUBTILE_INPUT_CHANNEL=16
        # parallel output channels.
        values = values.transpose(1, 0, 2, 3)
        weightTensor.values = _weightEncode(values.astype(np.uint8), bits, depthwise = True)
    weightTensor.name = f"{name}_{weightTensor.name}"

    return graph


@contextagnostic
class NE16AdjustWeightMemoryLayoutPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool, ne16EngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(
            graph,
            partial(_ne16_adjust_weight_memory_layout_fun,
                    default_channels_first = default_channels_first,
                    ne16EngineName = ne16EngineName), "_NE16_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS",
            NonBranchingMatcher(regex_op = True))


def _findAllMultiplicands(x: int) -> List[int]:
    multiplicands = []
    tmpX = x
    for i in range(2, math.ceil(math.sqrt(x))):  # Ceil cause range doesn't include the last number
        while tmpX % i == 0:
            multiplicands.append(i)
            tmpX = tmpX / i

    if x // math.prod(multiplicands) > 1:
        multiplicands.append(x // math.prod(multiplicands))

    return multiplicands


def _findAllReshapeOptions(dim: int) -> Generator[Tuple[int, int], None, None]:
    multiplicands = _findAllMultiplicands(dim)
    for combLen in range(1, 1 + (len(multiplicands) // 2)):
        for comb in itertools.combinations(multiplicands, combLen):
            a = math.prod(comb)
            b = dim // a
            yield a, b


def _nSubtiles(dims: Tuple[int, int]):
    return math.ceil(dims[0] / 6) * math.ceil(dims[1] / 6)


def _findLowestNumberOfSubtilesReshapeOptions(dim: int) -> List[Tuple[int, int]]:
    lowestNumberOfSubtiles = dim
    bestOptions: List[Tuple[int, int]] = [(dim, 1)]
    for option in _findAllReshapeOptions(dim):
        nSubtiles = _nSubtiles(option)
        if nSubtiles < lowestNumberOfSubtiles:
            lowestNumberOfSubtiles = nSubtiles
            bestOptions = [option]
        elif nSubtiles == lowestNumberOfSubtiles:
            bestOptions.append(option)
    return bestOptions


def _bestReshapeOption(dim: int) -> Tuple[int, int]:
    smallestDim = dim
    biggestDim = 1
    for option in _findLowestNumberOfSubtilesReshapeOptions(dim):
        if option[0] < smallestDim:
            smallestDim = option[0]
            biggestDim = option[1]
        elif option[1] < smallestDim:
            smallestDim = option[1]
            biggestDim = option[0]
    return biggestDim, smallestDim


def _ne16_reshape_pointwise_convolution_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                            ne16EngineName: str):
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    if not ("engine" in node.attrs and node.attrs["engine"] == ne16EngineName):
        return graph

    if not (node.attrs["kernel_shape"] == [1, 1]):
        return graph

    if "channels_first" in node.attrs:
        channels_first = node.attrs["channels_first"]
    else:
        channels_first = default_channels_first

    def extractSpatialDims(shape: List[int]) -> List[int]:
        if channels_first:
            return shape[-2:]
        else:
            return shape[-3:-1]

    def replaceSpatialDims(shape: List[int], newSpatialDims: Tuple[int, int]) -> List[int]:
        if channels_first:
            return shape[:-2] + list(newSpatialDims)
        else:
            return shape[:-3] + list(newSpatialDims) + shape[-1:]

    _input = node.inputs[0]
    spatialDims = extractSpatialDims(_input.shape)
    newSpatialDims = _bestReshapeOption(math.prod(spatialDims))
    newInputShape = replaceSpatialDims(_input.shape, newSpatialDims)

    inputReshapeNode, reshapedInput = _createReshape(_input, name, newInputShape)
    graph.nodes.append(inputReshapeNode)
    node.inputs[0] = reshapedInput

    output = node.outputs[0]
    newOutputShape = replaceSpatialDims(output.shape, newSpatialDims)
    reshapedOutput = gs.Variable(output.name + "_Reshaped", dtype = output.dtype, shape = newOutputShape)
    outputReshapeNode, _ = _createReshape(reshapedOutput, name, output.shape, output)
    graph.nodes.append(outputReshapeNode)
    node.outputs[0] = reshapedOutput

    return graph


@contextagnostic
class NE16ReshapePointwiseConvolutionPass(ReplaceSequentialPatternPass):
    """Reshape pointwise convolution's spatial dimensions so that they work better for N-EUREKA's hardware tiling"""

    def __init__(self, default_channels_first: bool, ne16EngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(
            graph,
            partial(_ne16_reshape_pointwise_convolution_fun,
                    default_channels_first = default_channels_first,
                    ne16EngineName = ne16EngineName), "_NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS",
            NonBranchingMatcher(regex_op = True))


class ConvEngineDiscolorationPass(EngineDiscolorationPass):

    def __init__(self):
        pattern = gs.Graph()
        _input = gs.Variable(name = 'input')
        output = pattern.layer(inputs = [_input], outputs = ['output'], op = 'RequantizedConv|Conv', name = 'conv')
        pattern.outputs.append(output)
        pattern.inputs.append(_input)
        super().__init__(pattern, "_CONV_ENGINE_DISCOLORATION_PASS", matcher = NonBranchingMatcher(regex_op = True))


@contextagnostic
class NE16OptimizationPass(SequentialPass):

    def __init__(self, default_channels_first: bool, ne16EngineName: str):
        super().__init__(NE16AdjustWeightMemoryLayoutPass(default_channels_first, ne16EngineName),
                         NE16ReshapePointwiseConvolutionPass(default_channels_first, ne16EngineName),
                         ReshapeMergePass(),
                         ReshapeConstOptPass(),
                         RemoveGlobalOutputReshapePass(),
                         name_prefix = '')
