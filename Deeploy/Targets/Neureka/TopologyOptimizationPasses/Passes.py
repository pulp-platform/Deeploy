# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
    NCHWtoNHWCConvPass, NCHWtoNHWCMaxPoolPass, NCHWtoNHWCPadPass, RemoveGlobalOutputReshapePass, _createReshape, \
    _isDepthwise, _NCWHtoNHWC_dw_fun, _PULP_NCHWtoNHWC_dw_fun, _singleNodePattern
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


def _spatialFactorPairs(n: int) -> Generator[Tuple[int, int], None, None]:
    """Yield every (a, b) with a * b == n and a >= b."""
    for b in range(1, math.isqrt(n) + 1):
        if n % b == 0:
            yield n // b, b


def _nSubtiles(height: int, width: int) -> int:
    """Number of 6x6 N-EUREKA HW subtiles needed to cover a (height, width) plane."""
    return math.ceil(height / 6) * math.ceil(width / 6)


def _bestSpatialReshape(n: int) -> Tuple[int, int]:
    """Find the (height, width) factorization of n needing the fewest 6x6 HW subtiles.

    Ties are broken toward the more square-like factorization, since that also
    tends to reduce padding waste in the border subtiles.
    """
    best = (n, 1)
    bestCost = _nSubtiles(*best)
    bestBalance = abs(best[0] - best[1])

    for candidate in _spatialFactorPairs(n):
        cost = _nSubtiles(*candidate)
        balance = abs(candidate[0] - candidate[1])
        if cost < bestCost or (cost == bestCost and balance < bestBalance):
            best, bestCost, bestBalance = candidate, cost, balance

    return best


def _extractSpatialDims(shape: List[int], channels_first: bool) -> List[int]:
    return shape[-2:] if channels_first else shape[-3:-1]


def _replaceSpatialDims(shape: List[int], newSpatialDims: Tuple[int, int], channels_first: bool) -> List[int]:
    if channels_first:
        return shape[:-2] + list(newSpatialDims)
    return shape[:-3] + list(newSpatialDims) + shape[-1:]


def _neureka_reshape_pointwise_convolution_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                               neurekaEngineName: str):
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    if not all([
            node.attrs.get("engine") == neurekaEngineName,
            node.attrs["kernel_shape"] == [1, 1],
    ]):
        return graph

    channels_first = bool(node.attrs.get("channels_first", default_channels_first))

    _input = node.inputs[0]
    output = node.outputs[0]

    inputSpatialDims = _extractSpatialDims(_input.shape, channels_first)
    outputSpatialDims = _extractSpatialDims(output.shape, channels_first)
    if math.prod(inputSpatialDims) != math.prod(outputSpatialDims):
        return graph

    newSpatialDims = _bestSpatialReshape(math.prod(inputSpatialDims))
    if tuple(inputSpatialDims) == newSpatialDims:
        return graph

    newInputShape = _replaceSpatialDims(_input.shape, newSpatialDims, channels_first)
    inputReshapeNode, reshapedInput = _createReshape(_input, name, newInputShape)
    graph.nodes.append(inputReshapeNode)
    node.inputs[0] = reshapedInput

    newOutputShape = _replaceSpatialDims(output.shape, newSpatialDims, channels_first)
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


def _neureka_nchw_to_nhwc_dw_conv_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                      neurekaEngineName: str) -> gs.Graph:
    node = next(iter(match.nodes_map.values()))

    if not _isDepthwise(node):
        return graph

    # DW convs have different data layouts depending on the engine that executes them:
    #   - N-EUREKA reads the input channels-last (NHWC) and the weight with the filter dimension last,
    #   - the PULP cluster kernel reads the input channels-first (NCHW) and the weight with the filter
    #     dimension first (see PULPOpen DWConvTileConstraint).
    # We dispatch on the engine the conv was colored with. This is authoritative here because the conv+requant
    # merge preserves the convolution's engine color (see PULPConvRequantMergePass), so the coloring interleaved
    # before this pass has already assigned the fused RequantizedConv to the correct engine.
    if node.attrs.get("engine") == neurekaEngineName:
        return _NCWHtoNHWC_dw_fun(graph, match, name, default_channels_first)
    return _PULP_NCHWtoNHWC_dw_fun(graph, match, name, default_channels_first)


@contextagnostic
class NeurekaNCHWtoNHWCDwConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool, neurekaEngineName: str):
        graph = _singleNodePattern(op = "RequantizedConv|Conv")
        name = "_NEUREKA_NCHW_TO_NHWC_DW_CONV_PASS"
        super().__init__(
            graph,
            partial(_neureka_nchw_to_nhwc_dw_conv_fun,
                    default_channels_first = default_channels_first,
                    neurekaEngineName = neurekaEngineName), name, NonBranchingMatcher(regex_op = True))


@contextagnostic
class NeurekaNCHWtoNHWCPass(SequentialPass):
    """Channels-last lowering pass for the N-EUREKA pipeline.

    Behaves like PULPNCHWtoNHWCPass/NCHWtoNHWCPass for pads, maxpools and regular convolutions, but lowers each
    depthwise convolution with the layout expected by the engine that will execute it (N-EUREKA or PULP cluster).
    """

    def __init__(self, default_channels_first: bool, neurekaEngineName: str):
        passes = [
            NCHWtoNHWCPadPass(default_channels_first),
            NCHWtoNHWCMaxPoolPass(default_channels_first),
            NeurekaNCHWtoNHWCDwConvPass(default_channels_first, neurekaEngineName),
            NCHWtoNHWCConvPass(default_channels_first),
        ]
        super().__init__(*passes)


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
