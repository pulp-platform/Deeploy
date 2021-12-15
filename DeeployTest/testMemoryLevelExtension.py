# ----------------------------------------------------------------------
#
# File: testMemoryLevelExtension.py
#
# Last edited: 04.05.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, ETH Zurich
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

import copy
import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.platformMapping import defaultScheduler, mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser, getPaths
from testUtils.typeMapping import inferInputType

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    NCHWtoNHWCPass, TransposeMatmulInputsPass
from Deeploy.DeeployTypes import StructBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper, \
    MemoryLevelAwareSignPropDeployer
from Deeploy.Targets.CortexM.Platform import CMSISEngine, CMSISMapping, CMSISOptimizer, CMSISPlatform
from Deeploy.Targets.Generic.Platform import GenericEngine, GenericMapping, GenericOptimizer, GenericPlatform
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import TransposeConstOptPass, TransposeMergePass
from Deeploy.Targets.MemPool.Platform import MemPoolEngine, MemPoolMapping, MemPoolOptimizer, MemPoolPlatform
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine, PULPMapping, PULPOptimizer, PULPPlatform

if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(description = "Test Utility for the Memory Level Extension.")
    args = parser.parse_args()

    inputTypes = {}
    inputOffsets = {}

    _GENDIRROOT = f'TEST_{args.platform.upper()}'
    _GENDIR, _TESTDIR, _TESTNAME = getPaths(args.dir, _GENDIRROOT)

    print("GENDIR    : ", _GENDIR)
    print("TESTDIR   : ", _TESTDIR)
    print("TESTNAME  : ", _TESTNAME)

    _DEEPLOYSTATEDIR = os.path.join(_GENDIR, "TEST_MEMORYLEVEL", "deeployStates")
    _DEEPLOYSTATEDIRMOCK = os.path.join(_GENDIR, "TEST_MEMORYLEVEL", "deeployStatesMock")

    onnx_graph = onnx.load_model(f'{_TESTDIR}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    inputs = np.load(f'{_TESTDIR}/inputs.npz')
    tensors = graph.tensors()

    # Load as int64 and infer types later
    test_inputs = [inputs[x].reshape(-1).astype(np.int64) for x in inputs.files]

    # Instantiate Classes Requried for Memory Level Annotation Extension
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 1024000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 512000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = 128000)

    memoryHierarchy = MemoryHierarchy([L3, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel("L3")
    defaultTargetMemoryLevel = L1

    platform, signProp = mapPlatform(args.platform)

    for index, num in enumerate(test_inputs):
        _type, offset = inferInputType(num, signProp)[0]
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset
        if "simpleRegression" in args.dir:
            inputOffsets[f"input_{index}"] = 0

    deployer = mapDeployer(platform, graph, inputTypes, deeployStateDir = _DEEPLOYSTATEDIR, inputOffsets = inputOffsets)

    if args.platform == "QEMU-ARM":

        MockEngine = CMSISEngine("MockCmsis", Mapping = copy.copy(CMSISMapping))
        MockPlatform = CMSISPlatform(engines = [MockEngine])
        MockPlatform = setupMemoryPlatform(MockPlatform, memoryHierarchy, defaultTargetMemoryLevel)

        mockDeployer = MemoryLevelAwareSignPropDeployer(graph,
                                                        MockPlatform,
                                                        inputTypes,
                                                        CMSISOptimizer,
                                                        defaultScheduler,
                                                        name = "DeeployNetwork",
                                                        deeployStateDir = _DEEPLOYSTATEDIR,
                                                        default_channels_first = False)

        # # Manually add the necessary optimization passes to parse WaveFormer
        mockDeployer.loweringOptimizer.passes += [
            TransposeMatmulInputsPass(),
            NCHWtoNHWCPass(deployer.default_channels_first),
            TransposeMergePass(),
            TransposeConstOptPass()
        ]

    elif args.platform == "MemPool":

        MockEngine = MemPoolEngine("MockMemPool", Mapping = copy.copy(MemPoolMapping))
        MockPlatform = MemPoolPlatform(engines = [MockEngine])
        MockPlatform = setupMemoryPlatform(MockPlatform, memoryHierarchy, defaultTargetMemoryLevel)

        mockDeployer = MemoryLevelAwareSignPropDeployer(graph,
                                                        MockPlatform,
                                                        inputTypes,
                                                        MemPoolOptimizer,
                                                        defaultScheduler,
                                                        name = "DeeployNetwork",
                                                        deeployStateDir = _DEEPLOYSTATEDIR,
                                                        default_channels_first = True)

    elif args.platform == "Generic":

        MockEngine = GenericEngine("MockGeneric", Mapping = copy.copy(GenericMapping))
        MockPlatform = GenericPlatform(engines = [MockEngine])
        MockPlatform = setupMemoryPlatform(MockPlatform, memoryHierarchy, defaultTargetMemoryLevel)

        mockDeployer = MemoryLevelAwareSignPropDeployer(graph,
                                                        MockPlatform,
                                                        inputTypes,
                                                        GenericOptimizer,
                                                        defaultScheduler,
                                                        name = "DeeployNetworkMock",
                                                        deeployStateDir = _DEEPLOYSTATEDIRMOCK,
                                                        default_channels_first = True)

    elif args.platform == "Siracusa":

        MockEngine = PULPClusterEngine("MockPulpCluster", Mapping = copy.copy(PULPMapping))
        MockPlatform = PULPPlatform(engines = [MockEngine])
        MockPlatform = setupMemoryPlatform(MockPlatform, memoryHierarchy, defaultTargetMemoryLevel)

        mockDeployer = MemoryLevelAwareSignPropDeployer(graph,
                                                        MockPlatform,
                                                        inputTypes,
                                                        PULPOptimizer,
                                                        defaultScheduler,
                                                        name = "DeeployNetworkMock",
                                                        deeployStateDir = _DEEPLOYSTATEDIRMOCK,
                                                        default_channels_first = False)

        # Manually add the necessary optimization pass to parse WaveFormer
        mockDeployer.loweringOptimizer.passes += [
            TransposeMatmulInputsPass(),
            NCHWtoNHWCPass(mockDeployer.default_channels_first),
            TransposeMergePass(),
            TransposeConstOptPass()
        ]

    else:
        raise RuntimeError(f"Deployment platform {args.platform} is not implemented")

    # Make the deployer memory-level aware
    deployer.Platform = setupMemoryPlatform(deployer.Platform, memoryHierarchy, defaultTargetMemoryLevel)
    deployer = MemoryDeployerWrapper(deployer)

    # Run the middleware and backend
    mockDeployer.generateFunction()
    deployer.generateFunction()

    # Test if the Contexts are correctly annotated for both the deployer and the mockDeployer
    defaultMemoryLevel = deployer.Platform.memoryHierarchy.getDefaultMemoryLevel()

    for bufferName, buffer in deployer.ctxt.globalObjects.items():
        if isinstance(buffer, VariableBuffer) and not isinstance(buffer, StructBuffer):
            assert buffer._memoryLevel == defaultMemoryLevel.name, f"Tensor {bufferName} in global scope of the deployer is not annotated with the default memory level"

    for bufferName, buffer in deployer.ctxt.localObjects.items():
        if isinstance(buffer, VariableBuffer) and not isinstance(buffer, StructBuffer):
            assert buffer._memoryLevel == defaultMemoryLevel.name, f"Tensor {bufferName} in local scope of the deployer is not annotated with the default memory level"

    for bufferName, buffer in mockDeployer.ctxt.globalObjects.items():
        if isinstance(buffer, VariableBuffer) and not isinstance(buffer, StructBuffer):
            assert buffer._memoryLevel == defaultMemoryLevel.name, f"Tensor {bufferName} in global scope of the mock deployer is not annotated with the default memory level"

    for bufferName, buffer in mockDeployer.ctxt.localObjects.items():
        if isinstance(buffer, VariableBuffer) and not isinstance(buffer, StructBuffer):
            assert buffer._memoryLevel == defaultMemoryLevel.name, f"Tensor {bufferName} in local scope of the mock deployer is not annotated with the default memory level"

    # Test if the memoryHierarchy attribute of the deployer and mockDeployer are equal
    assert mockDeployer.Platform.memoryHierarchy == deployer.Platform.memoryHierarchy, "Memory hierarchy of the deployer and mock deployer are not equal"

    # Test if the equality fails correctly if the memory hierarchy does not contain the same nodes
    L3 = MemoryLevel(name = "L3", neighbourNames = [], size = 1024000)
    memoryHierarchyAltered1 = MemoryHierarchy([L3])
    memoryHierarchyAltered1.setDefaultMemoryLevel("L3")
    mockDeployer.Platform.memoryHierarchy = memoryHierarchyAltered1
    assert not mockDeployer.Platform.memoryHierarchy == deployer.Platform.memoryHierarchy, "Memory hierarchy of the deployer and mock deployer are equal but are not supposed to be"

    # Test if the equality fails correctly if the default memory hierarchy is not the same
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 1024000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 752000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = 128000)
    memoryHierarchyAltered2 = MemoryHierarchy([L3, L2, L1])
    memoryHierarchyAltered2.setDefaultMemoryLevel("L3")
    mockDeployer.Platform.memoryHierarchy = memoryHierarchyAltered2
    assert not mockDeployer.Platform.memoryHierarchy == deployer.Platform.memoryHierarchy, "Memory hierarchy of the deployer and mock deployer are equal but are not supposed to be"

    print("Memory Level Extension test passed!")
