# ----------------------------------------------------------------------
#
# File: testEngineAwareOptimizerWrapper.py
#
# Last edited: 10.10.2023.
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
#   - Luka Macan, University of Bologna
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

import os

import onnx
import onnx_graphsurgeon as gs
from testUtils.graphColoring import graph_coloring
from testUtils.graphDebug import graphDiff

from Deeploy.DeeployTypes import TopologyOptimizationPass, TopologyOptimizer
from Deeploy.DeploymentEngine import EngineAwareOptimizerWrapper
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.BasicPasses import IntegerDivRequantMergePass, \
    MergeConstAddAndRequantPass, TransposeConstOptPass, TransposeMergePass, iGELURequantMergePass
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.CMSISPasses import ConvRequantMergePass, \
    GEMMRequantMergePass, LinearAttentionAlignmentPass, MatMulRequantMergePass, MHSAAlignmentPass
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.DebugPasses import DebugPrintMergePass
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import NCHWtoNHWCPass, \
    TransposeMatmulInputsPass
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.PULPPasses import PULPConvRequantMergePass


def _test_partial_coloring():
    test_dir = "Tests/simpleRegression"

    model = onnx.load(os.path.join(test_dir, "network.onnx"))
    graph = gs.import_onnx(model).toposort()

    graph = graph_coloring(graph, ["mockedEngine0", "mockedEngine1"], frequency = [3, 20], color_attr = "engine")

    original_optimizer = TopologyOptimizer([])
    optimizer = EngineAwareOptimizerWrapper(original_optimizer, engineName = "mockedEngine0")

    assert id(original_optimizer.passes) == id(
        optimizer.passes), "The wrapped optimizer does not expose original optimizers passes attribute."

    assert len(original_optimizer.passes) == 0
    optimizer.passes = [PULPConvRequantMergePass()]
    assert len(
        original_optimizer.passes) == 1, "Wrapped optimizer does not modify the original optimizers passes attribute."

    graph = optimizer.optimize(graph)
    graph = graph.cleanup().toposort()

    assert graph.nodes[
        0].op == "RequantizedConv", f"First 2 nodes should have been replaced with the RequantizedConv node. Got {graph.nodes[0].op}"

    non_optimized = [2, 5, 8]
    for i in non_optimized:
        assert graph.nodes[i].op == "Conv" and graph.nodes[
            i +
            1].op == "RequantShift", f"Nodes outside of the subgraph shouldn't have been optimized. Failed on nodes {i}:{graph.nodes[i].op} and {i+1}:{graph.nodes[i+1].op}"


def _test_pass(_pass: TopologyOptimizationPass, graph: gs.Graph, engineName: str) -> gs.Graph:
    # Mock Engine
    engineName = "mockEngine"

    # Mock coloring
    for node in graph.nodes:
        node.attrs["engine"] = engineName

    topologyOptimizer = TopologyOptimizer([_pass])
    engineOptimizer = EngineAwareOptimizerWrapper(topologyOptimizer, engineName)

    topologyOptimizedGraph = topologyOptimizer.optimize(graph.copy())

    # Mock recoloring
    for node in topologyOptimizedGraph.nodes:
        node.attrs["engine"] = engineName

    engineOptimizedGraph = engineOptimizer.optimize(graph.copy())

    diffTree = graphDiff(topologyOptimizedGraph, engineOptimizedGraph)
    assert diffTree.root is None, f"Failed at pass {type(_pass).__name__}\n{diffTree.message}"

    return topologyOptimizedGraph


def _test_passes():
    test_dir = "Tests/simpleRegression"
    model = onnx.load(os.path.join(test_dir, "network.onnx"))
    graph = gs.import_onnx(model).toposort()
    passes = [
        IntegerDivRequantMergePass(),
        iGELURequantMergePass(),
        LinearAttentionAlignmentPass(),
        MHSAAlignmentPass(),
        MergeConstAddAndRequantPass(),
        ConvRequantMergePass(),
        GEMMRequantMergePass(),
        MatMulRequantMergePass(),
        TransposeMatmulInputsPass(),
        NCHWtoNHWCPass(False),
        TransposeMergePass(),
        TransposeConstOptPass(),
        DebugPrintMergePass()
    ]

    for _pass in passes:
        graph = _test_pass(_pass, graph, "mockEngine")


if __name__ == "__main__":
    _test_partial_coloring()
    _test_passes()
    print("Test passed")
