# ----------------------------------------------------------------------
#
# File: Passes.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    """Unroll weight into expected memory format

    Expected weight shape is (cout, cin, H, W).
    The produced memory layout depends on the weight kernel shape:
      - 3x3: (cout, cinMajor, Bits, H x W x cinMinor_3x3 packed into Weight Bandwidth bits),
      - 1x1: (cout, cinMajor, Bits x H x W x cinMinor_1x1 packed into Weight Bandwidth bits),
    where cinMajor is the ceil(cin / cin subtile <mode>) and cinMinor has to be padded with 0 to cin subtile <mode>.
    """
    _NEUREKA_WEIGHT_BANDWIDTH = 256
    _NEUREKA_CIN_SUBTILE_1x1 = 32
    _NEUREKA_CIN_SUBTILE_3x3 = 28

    if depthwise:
        weight = weight.transpose(1, 0, 2, 3)  # Swap cout and cin

    cout, cin, height, width = weight.shape
    cinSubtile = (_NEUREKA_CIN_SUBTILE_3x3 if height == 3 else _NEUREKA_CIN_SUBTILE_1x1)

    # Pad cin to be divisible with CIN_SUBTILE
    if cin % cinSubtile != 0:
        cinPad = cinSubtile - cin % cinSubtile
        weight = np.pad(
            weight,
            ((0, 0), (0, cinPad), (0, 0), (0, 0)),
            "constant",
            constant_values = 0,
        )

    # Reshape into (cout, cinMajor, cinMinor, Flattened spatial, 1)
    # The 1 at the end is required by the unpacking
    cinMajor = int(np.ceil(cin / cinSubtile))
    weight = weight.reshape(cout, cinMajor, cinSubtile, height * width, 1)

    # Unpack 'bits' bits in little order, e.g. bits=4: 3 => [1, 1, 0, 0]
    # (cout, cinMajor, cinSubtile, Flattened spatial, Bits)
    weight = np.unpackbits(weight, axis = -1, count = bits, bitorder = "little")

    # Shuffle bits so that the final shape is:
    # (cout, cinMajor, Bits, Flattened spatial, cinSubtile)
    weight = weight.transpose(0, 1, 4, 3, 2)

    # Pack dimensions to fit into weight bandwidth
    if height == 3 and width == 3:
        # (cout * cinMajor * Bits, H * W * cinSubtile)
        weight = weight.reshape(-1, height * width * cinSubtile)
        # Pad only the last dimension to weight bandwidth size
        # (-1, Weight Bandwidth)
        weight = np.pad(
            weight,
            ((0, 0), (0, _NEUREKA_WEIGHT_BANDWIDTH - weight.shape[-1])),
            "constant",
            constant_values = 0,
        )
    elif height == 1 and width == 1:
        # Tile cinSubtile into tiles of size 4
        # (cout, cinMajor, Bits, Flattened spatial, cinSubtileMajor, cinSubtileTile)
        weight = weight.reshape(cout, cinMajor, bits, height * width, cinSubtile // 4,
                                4)  # cout, cinMajor, bits, 1, 8, 4
        # Pad bits to 8
        if bits < 8:
            # (cout, cinMajor, PaddedBits, Flattened spatial, cinSubtileMajor, cinSubtileTile)
            weight = np.pad(
                weight,
                ((0, 0), (0, 0), (0, 8 - bits), (0, 0), (0, 0), (0, 0)),
                mode = "constant",
                constant_values = 0,
            )
        # (cout, cinMajor, Flattened spatial, cinSubtileMajor, PaddedBits, cinSubtileTile)
        weight = weight.transpose(0, 1, 3, 4, 2, 5)
        # (-1, Weight Bandwidth)
        weight = weight.reshape(cout * cinMajor, _NEUREKA_WEIGHT_BANDWIDTH)  # cout*cinMajor, 256b

    # Prepare for packing
    # (-1, Weight Bandwidth Bytes, 8)
    weightBandwidthBytes = int(np.ceil(_NEUREKA_WEIGHT_BANDWIDTH / 8))
    weight = np.stack(np.split(weight, weightBandwidthBytes, axis = -1), axis = -2)

    # Pack bits
    # (-1, Weight Bandwidth Bytes)
    weight = np.packbits(weight, axis = -1, bitorder = "little")

    if height == 1 and width == 1:
        # (cout, cinMajor, Weight Bandwidth Bytes)
        return weight.reshape(cout, cinMajor, weightBandwidthBytes)
    elif depthwise:
        return weight.reshape(cout, cinMajor, bits, weightBandwidthBytes)
    else:
        return weight.reshape(cout, cinMajor, bits, weightBandwidthBytes)


def _neureka_adjust_weight_memory_layout_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                             neurekaEngineName: str):
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    if not ("engine" in node.attrs and node.attrs["engine"] == neurekaEngineName):
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

    # Weight encode expects channels first
    if not channels_first:
        values = values.transpose(0, 3, 1, 2)

    bits = 8  # Support only 8 bit weights for now
    if node.attrs['group'] == 1:
        weightTensor.values = _weightEncode(values.astype(np.uint8), bits, depthwise = False)
    else:
        weightTensor.values = _weightEncode(values.astype(np.uint8), bits, depthwise = True)
    weightTensor.name = f"{name}_{weightTensor.name}"

    return graph


@contextagnostic
class NeurekaAdjustWeightMemoryLayoutPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool, neurekaEngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(
            graph,
            partial(_neureka_adjust_weight_memory_layout_fun,
                    default_channels_first = default_channels_first,
                    neurekaEngineName = neurekaEngineName), "_NEUREKA_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS",
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


def _neureka_reshape_pointwise_convolution_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                               neurekaEngineName: str):
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    if not ("engine" in node.attrs and node.attrs["engine"] == neurekaEngineName):
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
class NeurekaReshapePointwiseConvolutionPass(ReplaceSequentialPatternPass):
    """Reshape pointwise convolution's spatial dimensions so that they work better for N-EUREKA's hardware tiling"""

    def __init__(self, default_channels_first: bool, neurekaEngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(
            graph,
            partial(_neureka_reshape_pointwise_convolution_fun,
                    default_channels_first = default_channels_first,
                    neurekaEngineName = neurekaEngineName), "_NEUREKA_RESHAPE_POINTWISE_CONVOLUTION_PASS",
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
class NeurekaOptimizationPass(SequentialPass):

    def __init__(self, default_channels_first: bool, neurekaEngineName: str):
        super().__init__(NeurekaAdjustWeightMemoryLayoutPass(default_channels_first, neurekaEngineName),
                         NeurekaReshapePointwiseConvolutionPass(default_channels_first, neurekaEngineName),
                         ReshapeMergePass(),
                         ReshapeConstOptPass(),
                         RemoveGlobalOutputReshapePass(),
                         name_prefix = '')
