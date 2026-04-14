# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _compute_ne16_scale_shift(mul_values, log2D):
    """Convert Deeploy's mul/log2D to NE16's per-channel scale/scale_n."""
    Ko = len(mul_values)
    ne16_scale = np.zeros(Ko, dtype = np.uint8)
    ne16_scale_n = np.zeros(Ko, dtype = np.uint8)
    for ko in range(Ko):
        sf = float(mul_values[ko]) / float(2**log2D)
        if sf >= 1.0:
            sn = 0
            sc = min(255, max(1, int(round(sf))))
        elif sf > 0:
            sn = min(31, max(0, int(math.floor(math.log2(127.0 / sf)))))
            sc = min(255, max(1, int(round(sf * (1 << sn)))))
        else:
            sn = 0
            sc = 0
        ne16_scale[ko] = sc
        ne16_scale_n[ko] = sn
    return ne16_scale, ne16_scale_n


def _ne16_adjust_gemm_weight_layout_fun(graph: gs.Graph, match: Match, name: str):
    """Prepare GEMM node for NE16 execution.

    Handles transB normalization, scale/scale_n computation, and bias rescaling.
    Weight bitplane packing and signed bias compensation are deferred to alignToContext
    where input signedness is known from the type system.
    """
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    # Weight is input[1] for both Gemm and RequantizedGemm
    weightTensor = node.inputs[1]

    if not isinstance(weightTensor, gs.Constant):
        return graph

    values = weightTensor.values

    # Skip true float weights (Deeploy stores int8 weights as float32)
    if not np.array_equal(values, np.round(values)):
        return graph

    # Check shape is 2D
    if len(values.shape) != 2:
        return graph

    # Determine actual Ko, Ki based on transB
    transB = node.attrs.get('transB', 0)
    if transB:
        Ko, Ki = values.shape
    else:
        Ki, Ko = values.shape

    # Check NE16 compatibility BEFORE modifying the node
    if Ki % 16 != 0:
        return graph

    # Transpose weight to [Ko, Ki] if needed — keep as int8
    if not transB:
        transposed = values.T.astype(np.int8)
        newWeightTensor = gs.Constant(f"{name}_{weightTensor.name}", transposed)
        node.inputs[1] = newWeightTensor
        node.attrs['transB'] = 1

    # For RequantizedGemm: transform mul → ne16_scale, create scale_n, rescale bias
    if node.op == 'RequantizedGemm' and len(node.inputs) >= 4:
        mulTensor = node.inputs[3]
        biasTensor = node.inputs[2]

        if isinstance(mulTensor, gs.Constant) and isinstance(biasTensor, gs.Constant):
            mul_values = mulTensor.values.flatten().astype(np.int32)
            log2D = int(np.log2(node.attrs['div'].values))

            # Broadcast scalar mul to per-channel if needed
            if len(mul_values) == 1:
                mul_values = np.full(Ko, mul_values[0], dtype = np.int32)

            ne16_scale, ne16_scale_n = _compute_ne16_scale_shift(mul_values, log2D)

            # Rescale bias from mul/log2D domain to scale/scale_n domain
            # bias_merged is already *= mul from PULPGEMMRequantMergePass
            # NE16 needs: bias_ne16 = bias_merged * 2^(scale_n - log2D)
            bias_values = biasTensor.values.flatten().astype(np.int64)
            ne16_bias = np.zeros(Ko, dtype = np.int64)
            for ko in range(Ko):
                shift_diff = int(ne16_scale_n[ko]) - log2D
                if shift_diff >= 0:
                    ne16_bias[ko] = bias_values[ko] << shift_diff
                else:
                    ne16_bias[ko] = bias_values[ko] >> (-shift_diff)

            ne16_bias = ne16_bias.astype(np.int32)

            # Overwrite mul tensor with ne16_scale
            mulTensor.values = ne16_scale

            # Overwrite bias tensor
            biasTensor.values = ne16_bias

            # Append scale_n as new input[4]
            scale_n_tensor = gs.Constant(f"{name}_scale_n", ne16_scale_n)
            node.inputs.append(scale_n_tensor)

    return graph


@contextagnostic
class NE16AdjustGEMMWeightLayoutPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedGemm|Gemm', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(graph, _ne16_adjust_gemm_weight_layout_fun, "_NE16_ADJUST_GEMM_WEIGHT_LAYOUT_PASS",
                         NonBranchingMatcher(regex_op = True))
