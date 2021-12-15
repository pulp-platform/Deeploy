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
from typing import Dict, List

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.typeMapping import inferInputType

from Deeploy.DeeployTypes import NetworkContext, NetworkDeployer, ONNXLayer, Schedule, StructBuffer, TransientBuffer, \
    VariableBuffer
from Deeploy.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, AnnotateIOMemoryLevel
from Deeploy.TilingExtension.MemoryScheduler import MemoryBlock
from Deeploy.TilingExtension.TilerExtension import TilerDeployerWrapper, TilingSolution


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


def validateTilingTopologySolution(schedule: Schedule, tilingSchedule: Schedule, memoryHierarchy: MemoryHierarchy):

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
                if len(tensor.memoryConstraints.keys()) > 1
            }

            intermediateTensors = patternTensors - borderTensors
            assert intermediateTensors == ((usedTensors & producedTensors) |
                                           transientTensors), "ERROR in tilingSchedule!"

            assert borderTensors == (usedTensors - producedTensors) | (producedTensors -
                                                                       usedTensors), "ERROR in tilingSchedule!"

            l1Occupation = getMemoryOccupation(ctxt, tilingStep.tensorMemoryConstraints, "L1")
            assert l1Occupation <= memoryHierarchy.memoryLevels['L1'].size, "L1 usage is too high"

            l2Occupation = getMemoryOccupation(ctxt, tilingStep.tensorMemoryConstraints, "L2")
            assert l2Occupation <= memoryHierarchy.memoryLevels['L2'].size, "L2 usage is too high!"


def _findBlocks(memoryMap: Dict[str, List[List[MemoryBlock]]], name: str) -> List[MemoryBlock]:

    res = []

    for key, patterns in memoryMap.items():
        for pattern in patterns:
            for block in pattern:
                if block.name == name:
                    res.append(block)

    return res


def validateStaticMemoryLayoutSolution(ctxt: NetworkContext, memoryMap: Dict[str, List[List[MemoryBlock]]]):

    # SCHEREMO: Assert that every VariableBuffer and ConstantBuffer is fully allocated somewhere
    # SCHEREMO: This doesn't need to hold for depth-first tiling!
    for key, buf in {**ctxt.localObjects}.items():
        if not isinstance(buf, (VariableBuffer)) or isinstance(buf, TransientBuffer) or isinstance(buf, StructBuffer):
            continue

        # SCHEREMO: Exception for memory arenas
        if buf._users == []:
            continue

        blocks = _findBlocks(memoryMap, key)

        if len(blocks) == 0:
            raise Exception(f"Didn't find any allocation of {key}")

        buf = ctxt.lookup(key)
        blockSize = np.prod(buf.shape) * (buf._type.referencedType.typeWidth // 8)

        blockFound = False
        for block in blocks:
            size = block.addrSpace[1] - block.addrSpace[0]
            blockFound |= (size == blockSize)

        assert blockFound, f"Didn't find full allocation of block {key}, expected {size} got {blockSize}"


def validateDynamicMemoryLayoutSolution(ctxt: NetworkContext, tilingSchedule: TilingSolution,
                                        memoryMap: Dict[str, List[List[MemoryBlock]]]):

    # SCHEREMO: Assert that tilingSchedule is implemented
    for patternIdx, patternConstraints in enumerate(tilingSchedule):
        for nodeConstraint in patternConstraints.nodeConstraints:
            for tensorConstraint in nodeConstraint.tensorMemoryConstraints.values():

                blocks = _findBlocks(memoryMap, tensorConstraint.tensorName)
                blockLevels = [block.level for block in blocks]

                buf = ctxt.lookup(tensorConstraint.tensorName)

                for memoryConstraint in tensorConstraint.memoryConstraints.values():

                    # SCHEREMO: Don't check static allocation
                    if buf._memoryLevel == memoryConstraint.memoryLevel:
                        continue

                    assert memoryConstraint.memoryLevel in blockLevels, f"No constraint for {tensorConstraint.tensorName} memoryLevel {memoryConstraint.memoryLevel}"

                    patternBlocks = memoryMap[memoryConstraint.memoryLevel][patternIdx]

                    _block = [block for block in patternBlocks if block.name == tensorConstraint.tensorName]

                    assert len(_block) == 1, f"{tensorConstraint.tensorName} not exactly once in pattern {patternIdx}!"

                    block = _block[0]
                    otherBlocks = [oblock for oblock in patternBlocks if oblock != block]

                    collisions = []
                    _buffer = ctxt.lookup(block.name)
                    for other in otherBlocks:
                        _otherBuffer = ctxt.lookup(other.name)
                        if (hasattr(_buffer, "_alias")
                                and _buffer._alias == other.name) or (hasattr(_otherBuffer, "_alias")
                                                                      and _otherBuffer._alias == block.name):
                            collisions.append(False)
                            continue

                        collisions.append(block.collides(other))

                    assert not any(collisions), f"{block.name} collides with another block in pattern {patternIdx}"

                    ctxtSize = memoryConstraint.size * memoryConstraint.multiBufferCoefficient * (
                        buf._type.referencedType.typeWidth // 8)
                    blockSize = block.addrSpace[1] - block.addrSpace[0]

                    assert ctxtSize <= blockSize, f"{tensorConstraint.tensorName}'s expected size does not match!"


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

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets,
                           scheduler = _mockScheduler)

    memoryLevelAnnotationPasses = [AnnotateDefaultMemoryLevel(memoryHierarchy), AnnotateIOMemoryLevel("L2")]

    # Make the deployer memory-level aware
    deployer.Platform = setupMemoryPlatform(deployer.Platform,
                                            memoryHierarchy,
                                            defaultTargetMemoryLevel = memoryHierarchy.memoryLevels["L1"])
    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer, memoryLevelAnnotationPasses)

    # Make the deployer tiler aware
    deployer = TilerDeployerWrapper(deployer)

    deployer.frontEnd()
    #deployer.midEnd()

    return deployer


def validateEffectiveLoad(outerMemoryMap: Dict[str, List[List[MemoryBlock]]],
                          innerMemoryMap: Dict[str, List[List[MemoryBlock]]], memoryHierarchy: MemoryHierarchy):
    staticLoadDict = {}
    maxAddr = 0
    for level, patterns in outerMemoryMap.items():
        maxAddr = 0
        for pattern in patterns:
            for block in pattern:
                maxAddr = max(maxAddr, block.addrSpace[1])
        staticLoadDict[level] = maxAddr

    dynamicLoadDict = {}
    maxAddr = 0
    for level, patterns in innerMemoryMap.items():
        maxAddr = 0
        for pattern in patterns:
            for block in pattern:
                maxAddr = max(maxAddr, block.addrSpace[1])
        dynamicLoadDict[level] = maxAddr

    totalLoadDict = {}
    for level in dynamicLoadDict.keys():
        totalLoadDict[level] = staticLoadDict[level] + dynamicLoadDict[level]

    for level, load in totalLoadDict.items():
        assert memoryHierarchy.memoryLevels[
            level].size > load, f"Effective memory layout does not fit {memoryHierarchy.memoryLevels[level].size} in {level}"


def validateDynamicLifetimes(ctxt: NetworkContext, tilingSchedule: TilingSolution,
                             outerMemoryMap: Dict[str, List[List[MemoryBlock]]]):

    for patternIdx, pattern in enumerate(tilingSchedule):
        for nodeConstraint in pattern.nodeConstraints:
            for tensor in nodeConstraint.tensorMemoryConstraints.values():
                name = tensor.tensorName

                buf = ctxt.lookup(name)
                if isinstance(buf, TransientBuffer) or ctxt.is_global(name):
                    continue

                blocks = _findBlocks(outerMemoryMap, name)
                assert len(blocks) == 1, f"Found {name} more than once in static life time map!"

                block = blocks[0]
                assert (patternIdx >= block.lifetime[0] and patternIdx
                        <= block.lifetime[1]), f"Tile of {name} is used after deallocation of the static buffer!"


if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(description = "Test Utility for the Scheduling Extension.")
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
    #memoryHierarchy.setDefaultMemoryLevel("L3.1")
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

        validateTilingTopologySolution(schedule, tilingSchedule, memoryHierarchy)

        innerMemoryMap = deployer.tiler.innerMemoryScheduler.memoryMap
        outerMemoryMap = deployer.tiler.outerMemoryScheduler.memoryMap

        validateStaticMemoryLayoutSolution(ctxt, outerMemoryMap)
        validateDynamicMemoryLayoutSolution(ctxt, tilingSchedule, innerMemoryMap)

        validateDynamicLifetimes(ctxt, tilingSchedule, outerMemoryMap)

        validateEffectiveLoad(outerMemoryMap, innerMemoryMap, memoryHierarchy)

        print("Tiler test ended, no memory violations!")
