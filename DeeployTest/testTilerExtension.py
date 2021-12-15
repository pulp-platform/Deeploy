# ----------------------------------------------------------------------
#
# File: tilerExtensionTest.py
#
# Last edited: 09.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
from typing import List

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.typeMapping import inferInputType

from Deeploy.DeeployTypes import GlobalDefinition, NetworkDeployer, ONNXLayer, Schedule, TransientBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, \
    AnnotateIOMemoryLevel
from Deeploy.TilingExtension.TilerExtension import TilerDeployerWrapper

# Mock of the Global Scheduler's inteface
# Returns a list of list of nodes instead of simply a list
# Inner list represent the patter over which we tile


def _mockScheduler(graph: gs.Graph) -> List[List[gs.Node]]:

    schedule = [[node] for node in graph.nodes]

    return schedule


def _filterSchedule(schedule: List[List[gs.Node]], layerBinding: 'OrderedDict[str, ONNXLayer]') -> List[List[gs.Node]]:

    filteredSchedule = []

    for pattern in schedule:

        filteredSchedulePattern = []
        for node in pattern:
            if node.name in layerBinding.keys():
                filteredSchedulePattern.append(node)
        filteredSchedule.append(filteredSchedulePattern)

    return filteredSchedule


def getMemoryOccupation(ctxt, tiledTensors, memoryLevel):

    occupation = 0

    for tensor in ctxt.globalObjects.values():

        if isinstance(tensor, GlobalDefinition):
            continue

        if tensor._memoryLevel == memoryLevel and tensor._users != []:
            occupation += np.prod(tensor.shape) * (tensor._type.referencedType.typeWidth // 8)

    for tensor in tiledTensors.values():
        for memoryConstraint in tensor.memoryConstraints.values():
            if memoryConstraint.memoryLevel == memoryLevel:

                if not isinstance(ctxt.lookup(tensor.tensorName), TransientBuffer):
                    typeWidth = (ctxt.lookup(tensor.tensorName)._type.referencedType.typeWidth // 8)
                else:
                    typeWidth = 1

                delta = memoryConstraint.multiBufferCoefficient * memoryConstraint.size * typeWidth
                occupation += delta

    return occupation


def validateSolution(schedule: Schedule, tilingSchedule: Schedule, memoryHierarchy: MemoryHierarchy):

    assert len(schedule) == len(tilingSchedule), "ERROR: schedule and tilingSchedule don't have the same length"

    for pattern, tilingPattern in zip(schedule, tilingSchedule):
        subGraph = gs.Graph(nodes = pattern)
        patternTensors = set([key for key, value in subGraph.tensors().items() if ctxt.lookup(key)._deploy])

        # intermediateTensors are all tensors that are used and produced by the pattern.
        # Including transient Buffers!
        usedTensors = set()
        producedTensors = set()
        transientTensors = set()

        for tensor in patternTensors:
            users = ctxt.lookup(tensor)._users

            for node in pattern:
                if node.name in users:
                    usedTensors.add(tensor)
                    break

        for node in pattern:
            outputTensors = {node.name for node in node.outputs}
            producedTensors |= outputTensors

        for tensorName, varBuffer in ctxt.localObjects.items():
            if isinstance(varBuffer, TransientBuffer):
                assert len(varBuffer._users) == 1
                if varBuffer._users[0] in patternTensors:
                    transientTensors.add(tensorName)

        for tilingStep in tilingPattern.nodeConstraints:
            borderTensors = {
                tensor.tensorName
                for tensor in tilingStep.tensorMemoryConstraints.values()
                if len(tensor.memoryConstraints) > 1
            }

            intermediateTensors = patternTensors - borderTensors

            assert intermediateTensors == ((usedTensors & producedTensors) |
                                           transientTensors), "ERROR in tilingSchedule!"
            assert borderTensors == (usedTensors - producedTensors) | (producedTensors -
                                                                       usedTensors), "ERROR in tilingSchedule!"

            l1Occupation = getMemoryOccupation(ctxt, tilingStep.tensorMemoryConstraints, "L1")
            assert l1Occupation <= memoryHierarchy.memoryLevels['L1'].size, "L1 usage is too high!"

            l2Occupation = getMemoryOccupation(ctxt, tilingStep.tensorMemoryConstraints, "L2")
            assert l2Occupation <= memoryHierarchy.memoryLevels['L2'].size, "L2 usage is too high!"


def setupDeployer(memoryHierarchy: MemoryHierarchy, graph: gs.Graph) -> NetworkDeployer:

    inputTypes = {}
    inputOffsets = {}

    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    inputs = np.load(f'./{args.dir}/inputs.npz')
    tensors = graph.tensors()

    # Load as int64 and infer types later
    test_inputs = [inputs[x].reshape(-1).astype(np.int64) for x in inputs.files]

    platform, signProp = mapPlatform(args.platform)

    for index, num in enumerate(test_inputs):
        _type, offset = inferInputType(num, signProp)[0]
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset
        if "simpleRegression" in args.dir:
            inputOffsets[f"input_{index}"] = 0

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets,
                           scheduler = _mockScheduler)

    memoryLevelAnnotationPasses = [AnnotateIOMemoryLevel("L2"), AnnotateDefaultMemoryLevel(memoryHierarchy)]

    # Make the platform memory-level aware
    deployer.Platform = setupMemoryPlatform(deployer.Platform,
                                            memoryHierarchy,
                                            defaultTargetMemoryLevel = memoryHierarchy.memoryLevels["L1"])
    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer, memoryLevelAnnotationPasses)

    # Make the deployer tiler aware
    deployer = TilerDeployerWrapper(deployer)

    deployer.frontEnd()

    return deployer


if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(description = "Test Utility for the Tiler Extension.")

    parser.add_argument('--l1', metavar = 'l1', dest = 'l1', type = int, default = 64000, help = 'Set L1 size\n')
    parser.add_argument('--shouldFail', action = 'store_true')
    parser.set_defaults(shouldFail = False)
    args = parser.parse_args()

    onnx_graph = onnx.load_model(f'./{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    # Instantiate Classes Requried for Memory Level Annotation Extension
    L3_2 = MemoryLevel(name = "L3.1", neighbourNames = ["L2"], size = 1024000)
    L3_1 = MemoryLevel(name = "L3.2", neighbourNames = ["L2"], size = 4000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3.1", "L3.2", "L1"], size = 512000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = args.l1)

    memoryHierarchy = MemoryHierarchy([L3_1, L3_2, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel("L2")

    deployer = setupDeployer(memoryHierarchy, graph)

    schedule = _filterSchedule(_mockScheduler(graph), deployer.layerBinding)

    if args.shouldFail:
        with pytest.raises(Exception):
            tilingSchedule = deployer.tiler.computeTilingSchedule(deployer.ctxt)

        print("Tiler test ended, failed as expected!")
    else:

        _ = deployer.generateFunction()

        tilingSchedule = deployer.tiler._getTilingSolution(deployer.tiler.tilerModel, deployer.ctxt,
                                                           deployer.tiler.tilerModel._collector,
                                                           deployer.tiler.symbolicMemoryConstraints)

        ctxt = deployer.ctxt
        layerBinding = deployer.layerBinding
        schedule = _mockScheduler(deployer.graph)

        validateSolution(schedule, tilingSchedule, memoryHierarchy)

        print("Tiler test ended, no memory violations!")
