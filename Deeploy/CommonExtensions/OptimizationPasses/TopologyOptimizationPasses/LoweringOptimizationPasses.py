# ----------------------------------------------------------------------
#
# File: LoweringOptimizationPasses.py
#
# Last edited: 07.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

from functools import partial
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, SequentialPass, \
    contextagnostic


def _createReshape(tensorIn: gs.Tensor,
                   name: str,
                   newShape: Sequence[Union[int, str]],
                   tensorOut: Optional[gs.Tensor] = None) -> Tuple[gs.Node, gs.Tensor]:
    newShapeConst = gs.Constant(name = name + tensorIn.name + "_NewShape", values = np.array(newShape))

    if tensorOut is None:
        tensorOut = gs.Variable(name = name + tensorIn.name + "_Reshaped", dtype = np.float32, shape = newShape)
    else:
        assert newShape == tensorOut.shape

    reshapeNode = gs.Node(name = name + tensorIn.name + "_Reshape",
                          op = "Reshape",
                          inputs = [tensorIn, newShapeConst],
                          outputs = [tensorOut])

    return reshapeNode, tensorOut


def _appendExpandDims(tensor: gs.Tensor, name: str, axis: Union[int, Sequence[int]]) -> Tuple[gs.Node, gs.Tensor]:
    if isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, tuple):
        axes = list(axis)
    elif isinstance(axis, list):
        axes = axis
    else:
        assert False, f"axis should be of type int or tuple. Got {type(axis)}"

    axes = [(len(tensor.shape) + len(axes) + axis if axis < 0 else axis) for axis in axes]
    assert all(axis >= 0 for axis in axes)

    assert isinstance(tensor.shape, Sequence) and len(tensor.shape) > 0 and isinstance(tensor.shape[0], int)
    assert all(axis < len(tensor.shape) + len(axes) for axis in axes), f"axis out of bounds. axis: {axes}"

    newShape = np.zeros(shape = (len(tensor.shape) + len(axes),), dtype = np.int_)
    for axis in axes:
        newShape[axis] = 1
    newShape[newShape == 0] = tensor.shape

    return _createReshape(tensor, name, newShape.tolist())


def _prependSqueezeDims(tensor: gs.Tensor, name: str, axis: Union[int, Sequence[int]]) -> Tuple[gs.Node, gs.Tensor]:
    if isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, tuple):
        axes = list(axis)
    elif isinstance(axis, list):
        axes = axis
    else:
        assert False, f"axis should be of type int or tuple. Got {type(axis)}"

    axes = [(len(tensor.shape) + axis if axis < 0 else axis) for axis in axes]
    assert all(axis >= 0 for axis in axes)

    assert isinstance(tensor.shape, Sequence) and len(tensor.shape) > 0 and isinstance(tensor.shape[0], int)
    assert all(axis < len(tensor.shape) + len(axes) for axis in axes), f"axis out of bounds. axis: {axes}"

    oldShape = np.zeros(shape = (len(tensor.shape) + len(axes),), dtype = np.int_)
    for axis in axes:
        oldShape[axis] = 1
    oldShape[oldShape == 0] = tensor.shape

    inputTensor = gs.Variable(name = name + tensor.name + "_Expanded", dtype = np.float32, shape = oldShape.tolist())

    reshapeNode, _ = _createReshape(inputTensor, name, tensor.shape, tensor)

    return reshapeNode, inputTensor


# Permute (0,1,2,3,...,N-2,N-1) -> (0,1,2,3,...,N-1,N-2)
def _permuteLastTwoDims(length: int) -> List[int]:
    outList = list(range(length))
    tmp = outList[-1]
    outList[-1] = outList[-2]
    outList[-2] = tmp
    return outList


# Permute (0,1,2,3,...,N-1) -> (0,2,3,...,N-1,1)
def _permuteNCHWtoNHWC(length: int) -> List[int]:
    outList = list(range(length))
    outList.remove(1)
    outList.append(1)
    return outList


# Permute (0,1,2,3,...,N-1) -> (0,N-1,1,2,3,...,N-2)
def _permuteNHWCtoNCHW(length: int) -> List[int]:
    outList = list(range(length))
    outList.remove(length - 1)
    outList.insert(1, length - 1)
    return outList


# Calculate permutation q = p^(-1) s.t. q(p(i)) = i
def _invertPermutation(permutation: List[int]) -> List[int]:
    tuples = []
    for idx, i in enumerate(permutation):
        tuples.append((i, idx))
    sortedTuples = sorted(tuples, key = lambda x: x[0])
    outPermutation = []
    for i in sortedTuples:
        outPermutation.append(i[1])
    return outPermutation


def _permuteList(inputList: List, permutation: List[int]):
    assert len(inputList) == len(permutation), "Permuted list and permutation must have equal length!"
    outList = []
    for i in permutation:
        outList.append(inputList[i])
    return outList


def _prependTransposeNode(anchor: gs.Variable,
                          nodeName: str,
                          permutation: Iterable[int],
                          invert: bool = False) -> (gs.Node, gs.Variable):

    if invert:
        outShape = _permuteList(anchor.shape, _invertPermutation(permutation))
    else:
        outShape = _permuteList(anchor.shape, permutation)

    anchorTransposeInput = gs.Variable(nodeName + "_Out", dtype = np.float32, shape = outShape)
    anchorTransposeNode = gs.Node(name = nodeName,
                                  op = "Transpose",
                                  inputs = [anchorTransposeInput],
                                  outputs = [anchor],
                                  attrs = {'perm': permutation})

    return anchorTransposeNode, anchorTransposeInput


def _appendTransposeNode(anchor: gs.Variable,
                         nodeName: str,
                         permutation: Iterable[int],
                         invert: bool = False) -> (gs.Node, gs.Variable):

    if invert:
        outShape = _permuteList(anchor.shape, _invertPermutation(permutation))
    else:
        outShape = _permuteList(anchor.shape, permutation)

    anchorTransposeOutput = gs.Variable(nodeName + "_In", dtype = np.float32, shape = outShape)
    anchorTransposeNode = gs.Node(name = nodeName,
                                  op = "Transpose",
                                  inputs = [anchor],
                                  outputs = [anchorTransposeOutput],
                                  attrs = {'perm': permutation})

    return anchorTransposeNode, anchorTransposeOutput


def _transposeMatMulInputs_fun(graph: gs.Graph, match: Match, name: str):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    gemmNode = matched_nodes[0]

    inputA = gemmNode.inputs[0]
    inputB = gemmNode.inputs[1]

    if 'transA' not in gemmNode.attrs:
        gemmNode.attrs['transA'] = 0
    if 'transB' not in gemmNode.attrs:
        gemmNode.attrs['transB'] = 0
    if 'alpha' not in gemmNode.attrs:
        gemmNode.attrs['alpha'] = 1.0
    if 'beta' not in gemmNode.attrs:
        gemmNode.attrs['beta'] = 1.0

    # Prepend transpose on A if it's transposed
    if gemmNode.attrs['transA'] != 0:
        anchorTransposeNode, anchorTransposeOutput = _appendTransposeNode(inputA, name + "_A",
                                                                          _permuteLastTwoDims(len(inputA.shape)))
        gemmNode.inputs[0] = anchorTransposeOutput
        gemmNode.attrs['transA'] = 0
        graph.nodes.append(anchorTransposeNode)

    # Prepend transpose on B if it's not transposed
    if gemmNode.attrs['transB'] != 1:
        anchorTransposeNode, anchorTransposeOutput = _appendTransposeNode(inputB, name + "_B",
                                                                          _permuteLastTwoDims(len(inputB.shape)))
        gemmNode.inputs[1] = anchorTransposeOutput
        gemmNode.attrs['transB'] = 1
        graph.nodes.append(anchorTransposeNode)

    return graph


# SCHEREMO:
# Implements generation of tranpose nodes such that each GEMM/MatMul node implements attributes transA = 0 transB = 1
@contextagnostic
class TransposeMatmulInputsPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['gemmOut'], op = 'RequantizedGemm', name = 'requantizedGemm')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_TRANSPOSE_MATMUL_INPUTS_PASS"
        super().__init__(graph, _transposeMatMulInputs_fun, name)


def _NCHWtoNHWC_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    opNode = matched_nodes[0]
    node_op = opNode.op

    # Default for non-existent channels_first: True
    channels_first = opNode.attrs["channels_first"] if "channels_first" in opNode.attrs else True

    if (channels_first != default_channels_first):

        inputNode = opNode.inputs[0]
        outputNode = opNode.outputs[0]

        inPermute = _permuteNCHWtoNHWC(len(inputNode.shape))
        outPermute = _permuteNHWCtoNCHW(len(outputNode.shape))

        inputTransposeNode, inputTransposeOutput = _appendTransposeNode(inputNode, name + "_TransposeIn", inPermute)
        outputTransposeNode, outputTransposeInput = _prependTransposeNode(outputNode,
                                                                          name + "_TransposeOut",
                                                                          outPermute,
                                                                          invert = True)

        opNode.inputs[0] = inputTransposeOutput
        opNode.outputs[0] = outputTransposeInput
        graph.nodes.append(inputTransposeNode)
        graph.nodes.append(outputTransposeNode)

        if node_op in ["RequantizedConv", "Conv"]:

            # Non DW-Type:
            if opNode.attrs['group'] == 1:
                weightNode = opNode.inputs[1]
                weightTransposeNode, weightTransposeOutput = _appendTransposeNode(weightNode, name + "TransposeWeight",
                                                                                  inPermute)

            else:
                DWPermute = [inPermute[-1]] + inPermute[1:-1] + [inPermute[0]]
                weightNode = opNode.inputs[1]
                weightTransposeNode, weightTransposeOutput = _appendTransposeNode(weightNode, name + "TransposeWeight",
                                                                                  DWPermute)

            opNode.inputs[1] = weightTransposeOutput
            graph.nodes.append(weightTransposeNode)

        opNode.attrs["channels_first"] = default_channels_first

    return graph


@contextagnostic
class NCHWtoNHWCMaxPoolPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['maxPool'], op = 'MaxPool', name = 'MaxPool')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_NCHW_TO_NHWC_MAXPOOL_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['convOut'], op = 'Conv', name = 'conv')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCRequantizedConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['convOut'], op = 'RequantizedConv', name = 'requantizedConv')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCPadPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['padOut'], op = 'Pad', name = 'pad')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_NCHW_TO_NHWC_PAD_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCPass(SequentialPass):

    def __init__(self, default_channels_first: bool = True):
        passes = [
            NCHWtoNHWCPadPass(default_channels_first),
            NCHWtoNHWCMaxPoolPass(default_channels_first),
            NCHWtoNHWCConvPass(default_channels_first),
            NCHWtoNHWCRequantizedConvPass(default_channels_first),
        ]
        super().__init__(*passes)


def _PULPDWNCHWtoNHWC_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    opNode = matched_nodes[0]
    node_op = opNode.op

    if opNode.attrs['group'] == 1:
        return graph

    if (("channels_first" in opNode.attrs and opNode.attrs["channels_first"] != default_channels_first)
            or ("channels_first" not in opNode.attrs and default_channels_first == 0)):

        inputNode = opNode.inputs[0]
        outputNode = opNode.outputs[0]

        inPermute = _permuteNCHWtoNHWC(len(inputNode.shape))
        outPermute = _permuteNHWCtoNCHW(len(outputNode.shape))

        outputTransposeNode, outputTransposeInput = _prependTransposeNode(outputNode,
                                                                          name + "_TransposeOut",
                                                                          outPermute,
                                                                          invert = True)

        opNode.outputs[0] = outputTransposeInput
        graph.nodes.append(outputTransposeNode)

        if node_op == "RequantizedConv":

            weightNode = opNode.inputs[1]
            weightTransposeNode, weightTransposeOutput = _appendTransposeNode(weightNode, name + "TransposeWeight",
                                                                              inPermute)
            opNode.inputs[1] = weightTransposeOutput
            graph.nodes.append(weightTransposeNode)

        opNode.attrs["channels_first"] = default_channels_first

    return graph


@contextagnostic
class PULPDWConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['convOut'], op = 'RequantizedConv', name = 'requantizedConv')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_PULPDWNCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


def _PULPDenseNCHWtoNHWC_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    opNode = matched_nodes[0]

    node_group = opNode.attrs['group'] if 'group' in opNode.attrs else 1
    if node_group != 1:
        return graph

    return _NCHWtoNHWC_fun(graph, match, name, default_channels_first)


@contextagnostic
class PULPNCHWtoNHWCDenseRequantizedConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['convOut'], op = 'RequantizedConv', name = 'requantizedConv')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_PULPDenseNCHWtoNHWC_fun, default_channels_first = default_channels_first),
                         name)


def _NeurekaDWNCHWtoNHWC_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    opNode = matched_nodes[0]

    node_group = opNode.attrs['group'] if 'group' in opNode.attrs else 1
    if node_group == 1:
        return graph

    return _NCHWtoNHWC_fun(graph, match, name, default_channels_first)


@contextagnostic
class NeurekaNCHWtoNHWCDWRequantizedConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['convOut'], op = 'RequantizedConv', name = 'requantizedConv')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_NeurekaDWNCHWtoNHWC_fun, default_channels_first = default_channels_first),
                         name)


@contextagnostic
class PULPNCHWtoNHWCDenseConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['convOut'], op = 'Conv', name = 'conv')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_PULPDenseNCHWtoNHWC_fun, default_channels_first = default_channels_first),
                         name)


@contextagnostic
class PULPNCHWtoNHWCPass(SequentialPass):

    def __init__(self, default_channels_first: bool = True):
        passes = [
            NCHWtoNHWCPadPass(default_channels_first),
            NCHWtoNHWCMaxPoolPass(default_channels_first),
            PULPDWConvPass(default_channels_first),
            PULPNCHWtoNHWCDenseConvPass(default_channels_first),
            PULPNCHWtoNHWCDenseRequantizedConvPass(default_channels_first),
        ]
        super().__init__(*passes)


@contextagnostic
class NeurekaNCHWtoNHWCPass(SequentialPass):

    def __init__(self, default_channels_first: bool = True):
        passes = [
            NCHWtoNHWCPadPass(default_channels_first),
            NCHWtoNHWCMaxPoolPass(default_channels_first),
            NeurekaNCHWtoNHWCDWRequantizedConvPass(default_channels_first),
            PULPNCHWtoNHWCDenseConvPass(default_channels_first),
            PULPNCHWtoNHWCDenseRequantizedConvPass(default_channels_first),
        ]
        super().__init__(*passes)


def _requantized_gemm_to_pw_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = list(match.nodes_map.values())
    requantizedGemm = matched_nodes[0]

    matrixA: gs.Variable = requantizedGemm.inputs[0]
    matrixB: gs.Constant = requantizedGemm.inputs[1]
    matrixY: gs.Variable = requantizedGemm.outputs[0]

    # Check matrixB is a constant, otherwise don't transform
    if not isinstance(matrixB, gs.Constant):
        return graph

    assert len(matrixA.shape) in [
        2, 3
    ], f"Unsupported number of dimensions for input matrix A of GEMM operation: {len(matrixA.shape)}; shape: {matrixA.shape}"
    assert len(matrixY.shape) in [
        2, 3
    ], f"Unsupported number of dimensions for output matrix of GEMM operation: {len(matrixY.shape)}; shape: {matrixY.shape}"

    # Pointwise with HWC layout (channels_first == False)

    # If transA is set then the matrix is of shape [B x K x M] and it needs to be transposed, otherwise its shape is  [B x M x K]
    if 'transA' in requantizedGemm.attrs and requantizedGemm.attrs['transA'] == 1:
        matrixATransposeNode, matrixA = _appendTransposeNode(matrixA, name, _permuteLastTwoDims(len(matrixA.shape)))
        graph.nodes.append(matrixATransposeNode)

    # Align dimensions for convolution
    expandAxis = []
    # Align the batch dimension
    if len(matrixA.shape) == 2:
        expandAxis.append(0)
    # Expand the height dimension
    expandAxis.append(1)
    # pwIn, shape [B x 1 x M x K]
    matrixAExpandDimsNode, pwIn = _appendExpandDims(matrixA, name, axis = expandAxis)
    graph.nodes.append(matrixAExpandDimsNode)

    # If transB is set then the matrix is of shape [N x K] and it doesn't need to be transposed, otherwise its shape is [K x N] and it has to be transposed
    if not 'transB' in requantizedGemm.attrs or requantizedGemm.attrs['transB'] == 0:
        # matrixBTransposed, shape [N x K]
        matrixBTransposeNode, matrixB = _appendTransposeNode(matrixB, name, _permuteLastTwoDims(len(matrixB.shape)))
        graph.nodes.append(matrixBTransposeNode)
    # pwWeight, shape [N x 1 x 1 x K]
    matrixBExpandDimsNode, pwWeight = _appendExpandDims(matrixB, name, axis = (1, 2))
    graph.nodes.append(matrixBExpandDimsNode)

    if len(matrixY.shape) == 2:
        # matrixY, shape [M x N]
        squeezeDims = (0, 1)
    else:
        # matrixY, shape [B x M x N]
        squeezeDims = (1,)
    # pwOut, shape [B x 1 x M x N]
    matrixYSqueezeDimsNode, pwOut = _prependSqueezeDims(matrixY, name, squeezeDims)
    graph.nodes.append(matrixYSqueezeDimsNode)

    pwAttrs = {
        'channels_first': False,
        'dilations': [1, 1],
        'group': 1,
        'kernel_shape': [1, 1],
        'pads': [0, 0, 0, 0],
        'strides': [1, 1],
        'div': requantizedGemm.attrs['div'],
        'n_levels_out': requantizedGemm.attrs['n_levels_out'],
        'shift': requantizedGemm.attrs['shift'],
        'signed': requantizedGemm.attrs['signed'],
    }

    add = requantizedGemm.inputs[2]
    mul = requantizedGemm.inputs[3]

    _inputs = [pwIn, pwWeight, mul, add]

    pw = gs.Node(op = 'RequantizedConv',
                 name = name + "_RequantizedPwConv",
                 inputs = _inputs,
                 outputs = [pwOut],
                 attrs = pwAttrs)
    graph.nodes.append(pw)

    requantizedGemm.inputs.clear()
    requantizedGemm.outputs.clear()
    graph.nodes.remove(requantizedGemm)

    return graph


@contextagnostic
class RequantizedGemmToPwPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedGemm', name = 'requantizedGemm')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(graph, _requantized_gemm_to_pw_fun, "_REQUANTIZED_GEMM_TO_PW_PASS")


def _remove_global_output_reshape_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = list(match.nodes_map.values())
    reshape = matched_nodes[0]

    isGlobalOutput = len(reshape.outputs[0].outputs) == 0

    if isGlobalOutput:
        graph.deleteNode(reshape)

    return graph


@contextagnostic
class RemoveGlobalOutputReshapePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'Reshape', name = 'reshape')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(graph, _remove_global_output_reshape_fun, "_REMOVE_GLOBAL_OUTPUT_RESHAPE_PASS")
