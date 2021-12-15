# ----------------------------------------------------------------------
#
# File: testMVP.py
#
# Last edited: 31.10.2023
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

import os
import random
from collections import OrderedDict
from typing import List, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from ortools.constraint_solver.pywrapcp import IntVar
from testUtils.codeGenerate import generateL3HexDump, generateTestInputsHeader, generateTestNetworkHeader, \
    generateTestNetworkImplementation, generateTestOutputsHeader
from testUtils.graphDebug import generateDebugConfig
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.typeMapping import inferInputType

from Deeploy.DeeployTypes import CodeGenVerbosity, ConstantBuffer, NetworkContext, NetworkDeployer, ONNXLayer, \
    SubGraph, TransientBuffer
from Deeploy.EngineExtension.NetworkDeployers.EngineColoringDeployer import EngineColoringDeployerWrapper
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, \
    AnnotateIOMemoryLevel, AnnotateNeurekaWeightMemoryLevel
from Deeploy.TilingExtension.MemoryScheduler import MemoryScheduler
from Deeploy.TilingExtension.TilerExtension import Tiler, TilerDeployerWrapper
from Deeploy.TilingExtension.TilerModel import TilerModel

_TEXT_ALIGN = 30


class DBOnlyL3Tiler(Tiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:

        varBuffer = ctxt.lookup(tensorName)

        generalCoeff = 2

        if isinstance(varBuffer, TransientBuffer):
            coefficient = 1
        elif isinstance(varBuffer, ConstantBuffer):
            coefficient = generalCoeff
        else:
            coefficient = generalCoeff

        if args.defaultMemLevel == "L2":
            return coefficient

        if hop == 'L1':
            return 1

        return coefficient


class DBTiler(Tiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        varBuffer = ctxt.lookup(tensorName)

        generalCoeff = 2

        if isinstance(varBuffer, TransientBuffer):
            coefficient = 1
        elif isinstance(varBuffer, ConstantBuffer):
            coefficient = generalCoeff
        else:
            coefficient = generalCoeff

        return coefficient


class SBTiler(DBTiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        varBuffer = ctxt.lookup(tensorName)

        generalCoeff = 1

        if isinstance(varBuffer, TransientBuffer):
            coefficient = 1
        elif isinstance(varBuffer, ConstantBuffer):
            coefficient = generalCoeff
        else:
            coefficient = generalCoeff

        return coefficient


class RandomizedMemoryScheduler(MemoryScheduler):

    def heuristicPermutation(self, adjacencyMatrix, costVector) -> List[int]:
        permutationList = list(range(len(costVector)))
        random.seed(self.seed)
        random.shuffle(permutationList)

        return permutationList


class RandomizedSBTiler(DBTiler):

    memorySchedulerClass = RandomizedMemoryScheduler

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        varBuffer = ctxt.lookup(tensorName)

        generalCoeff = 1

        if isinstance(varBuffer, TransientBuffer):
            coefficient = 1
        elif isinstance(varBuffer, ConstantBuffer):
            coefficient = generalCoeff
        else:
            coefficient = generalCoeff

        return coefficient


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


def setupDeployer(graph: gs.Graph,
                  memoryHierarchy: MemoryHierarchy,
                  defaultTargetMemoryLevel: MemoryLevel,
                  defaultIoMemoryLevel: MemoryLevel,
                  verbose: CodeGenVerbosity,
                  overwriteRecentState = False) -> NetworkDeployer:

    inputTypes = {}
    inputOffsets = {}

    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    inputs = np.load(f'{args.dir}/inputs.npz')
    tensors = graph.tensors()

    # Load as int64 and infer types later
    test_inputs = [inputs[x].reshape(-1).astype(np.int64) for x in inputs.files]

    platform, signProp = mapPlatform(args.platform)

    if args.enable_3x3:
        platform.engines[0].enable3x3 = True
    if args.enableStrides:
        platform.engines[0].enableStrides = True

    for index, num in enumerate(test_inputs):
        # WIESP: Do not infer types and offset of empty arrays
        if np.prod(num.shape) == 0:
            continue
        _type, offset = inferInputType(num, signProp)[0]
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets,
                           scheduler = _mockScheduler)

    # Make the deployer engine-color-aware
    if args.platform == "Siracusa_w_neureka":
        deployer = EngineColoringDeployerWrapper(deployer)

    # Make platform memory-aware after mapDeployer because it requires the platform to be an instance of an unwrapped platform
    deployer.Platform = setupMemoryPlatform(deployer.Platform, memoryHierarchy, defaultTargetMemoryLevel)

    memoryLevelAnnotationPasses = [
        AnnotateIOMemoryLevel(defaultIoMemoryLevel.name),
        AnnotateDefaultMemoryLevel(memoryHierarchy)
    ]

    if args.neureka_wmem:
        weightMemoryLevel = memoryHierarchy.memoryLevels["WeightMemory_SRAM"]
        memoryLevelAnnotationPasses.append(
            AnnotateNeurekaWeightMemoryLevel(neurekaEngineName = deployer.Platform.engines[0].name,
                                             weightMemoryLevel = weightMemoryLevel))

    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer, memoryLevelAnnotationPasses)

    # Make the deployer tiler aware
    if args.doublebuffer:
        deployer = TilerDeployerWrapper(deployer, DBOnlyL3Tiler)
    elif args.randomizedMemoryScheduler:
        deployer = TilerDeployerWrapper(deployer, RandomizedSBTiler)
    else:
        deployer = TilerDeployerWrapper(deployer, SBTiler)

    deployer.frontEnd()
    deployer.midEnd()

    # Decomposed Backend to mock the scheduler
    deployer.backEnd(verbose)

    deployer.prepared = True

    if overwriteRecentState:
        os.makedirs(f'./deeployStates/', exist_ok = True)
        os.system(f'cp -r {_DEEPLOYSTATEDIR}/* ./deeployStates/')

    return deployer


if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(
        description = "Deeploy Code Generation Utility with Memory Level Annotation and Tiling Extension.")

    parser.add_argument('--debug',
                        dest = 'debug',
                        action = 'store_true',
                        default = False,
                        help = 'Enable debugging mode\n')
    parser.add_argument('--defaultMemLevel',
                        metavar = 'defaultMemLevel',
                        dest = 'defaultMemLevel',
                        type = str,
                        default = "L2",
                        help = 'Set default memory level\n')
    parser.add_argument('--neureka-wmem',
                        dest = "neureka_wmem",
                        action = "store_true",
                        default = False,
                        help = 'Adds weight memory and neureka engine color\n')
    parser.add_argument('--enable-3x3',
                        dest = "enable_3x3",
                        action = "store_true",
                        default = False,
                        help = 'Adds EXPERIMENTAL support for 3x3 convolutions on N-EUREKA\n')
    parser.add_argument('--enableStrides',
                        dest = "enableStrides",
                        action = "store_true",
                        default = False,
                        help = 'Adds EXPERIMENTAL support for strided convolutions on N-EUREKA\n')
    parser.add_argument('--randomizedMemoryScheduler', action = "store_true")
    parser.add_argument('--doublebuffer', action = 'store_true')
    parser.add_argument('--l1', metavar = 'l1', dest = 'l1', type = int, default = 64000, help = 'Set L1 size\n')
    parser.add_argument('--shouldFail', action = 'store_true')
    parser.add_argument('--profileTiling',
                        metavar = 'profileTiling',
                        dest = 'profileTiling',
                        type = str,
                        default = None)
    parser.add_argument('--overwriteRecentState',
                        action = 'store_true',
                        help = 'Copy the recent deeply state to the ./deeployStates folder\n')

    parser.set_defaults(shouldFail = False)
    args = parser.parse_args()

    verbosityCfg = CodeGenVerbosity(None)

    if args.profileTiling is not None:
        verbosityCfg.tilingProfiling = args.profileTiling

    onnx_graph = onnx.load_model(f'{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    inputTypes = {}
    inputOffsets = {}

    inputs = np.load(f'{args.dir}/inputs.npz')
    outputs = np.load(f'{args.dir}/outputs.npz')
    if os.path.isfile(f'{args.dir}/activations.npz'):
        activations = np.load(f'{args.dir}/activations.npz')
    else:
        activations = None

    tensors = graph.tensors()

    if args.debug:
        test_inputs, test_outputs, graph = generateDebugConfig(inputs, outputs, activations, graph)
    else:
        # Load as int64 and infer types later
        test_inputs = [inputs[x].reshape(-1).astype(np.int64) for x in inputs.files]
        test_outputs = [outputs[x].reshape(-1).astype(np.int64) for x in outputs.files]

        # WIESEP: Hack to get CI running because only one specific array is used
        if "WaveFormer" in args.dir:
            test_inputs = [test_inputs[0]]
            test_outputs = [test_outputs[-2]]

    # Instantiate Classes Requried for Memory Level Annotation Extension
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 64000000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 512000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = args.l1)
    memoryLevels = [L3, L2, L1]

    if args.neureka_wmem:
        memoryLevels.append(MemoryLevel(name = "WeightMemory_SRAM", neighbourNames = [], size = 4 * 1024 * 1024))

    memoryHierarchy = MemoryHierarchy(memoryLevels)
    memoryHierarchy.setDefaultMemoryLevel(args.defaultMemLevel)

    deployer = setupDeployer(graph,
                             memoryHierarchy,
                             defaultTargetMemoryLevel = L1,
                             defaultIoMemoryLevel = memoryHierarchy.memoryLevels[args.defaultMemLevel],
                             verbose = verbosityCfg,
                             overwriteRecentState = args.overwriteRecentState)

    platform = deployer.Platform
    signProp = False

    for index, num in enumerate(test_inputs):
        # WIESP: Do not infer types and offset of empty arrays
        if np.prod(num.shape) == 0:
            continue
        _type, offset = inferInputType(num, signProp)[0]
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset

    schedule = _filterSchedule(_mockScheduler(graph), deployer.layerBinding)

    if args.shouldFail:
        with pytest.raises(Exception):
            tilingSchedule = deployer.tiler.computeTilingSchedule(deployer.ctxt)

        print("Tiler test ended, failed as expected!")
    else:

        _ = deployer.generateFunction(verbosityCfg)

        # Create input and output vectors
        os.makedirs(f'{args.dumpdir}', exist_ok = True)

        testInputStr = generateTestInputsHeader(deployer, test_inputs, inputTypes, inputOffsets)
        f = open(f'{args.dumpdir}/testinputs.h', "w")
        f.write(testInputStr)
        f.close()

        testOutputStr = generateTestOutputsHeader(deployer, test_outputs, signProp, args.verbose)
        f = open(f'{args.dumpdir}/testoutputs.h', "w")
        f.write(testOutputStr)
        f.close()

        # Generate code for Network
        testNetworkHeaderStr = generateTestNetworkHeader(deployer, platform)
        f = open(f'{args.dumpdir}/Network.h', "w")
        f.write(testNetworkHeaderStr)
        f.close()

        testNetworkImplementationStr = generateTestNetworkImplementation(deployer, platform)
        f = open(f'{args.dumpdir}/Network.c', "w")
        f.write(testNetworkImplementationStr)
        f.close()

        generateL3HexDump(deployer, os.path.join(f'{args.dumpdir}', 'hex'), test_inputs, test_outputs)

        clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
        os.system(f'clang-format -i --style="{clang_format}" {args.dumpdir}/Network.c')
        os.system(f'clang-format -i --style="{clang_format}" {args.dumpdir}/Network.h')
        os.system(f'clang-format -i --style="{clang_format}" {args.dumpdir}/testoutputs.h')
        os.system(f'clang-format -i --style="{clang_format}" {args.dumpdir}/testinputs.h')

        if args.verbose:
            print()
            print("=" * 80)
            num_ops = deployer.numberOfOps(args.verbose)
            print("=" * 80)
            print()
            print(f"{'Number of Ops:' :<{_TEXT_ALIGN}} {num_ops}")
            print('Worst Case Buffer Size:')
            for level in deployer.worstCaseBufferSize.keys():
                print(f"{'  ' + str(level) + ':' :<{_TEXT_ALIGN}} {deployer.worstCaseBufferSize[level]}")
            print(f"{'Model Parameters: ' :<{_TEXT_ALIGN}} {deployer.getParameterSize()}")

        print("Tiler test ended, no memory violations!")
