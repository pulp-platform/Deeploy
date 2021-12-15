# ----------------------------------------------------------------------
#
# File: deeployStateEqualityTest.py
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

import argparse
import copy
import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.typeMapping import inferInputType

from Deeploy.DeeployTypes import NetworkContext, StructBuffer, VariableBuffer, _backendPostBindingFilename, \
    _middlewarePreLoweringFilename
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Test Utility for the State Equality Check.")
    parser.add_argument('-t',
                        metavar = 'testdir',
                        dest = 'dir',
                        type = str,
                        default = './Tests/simpleRegression',
                        help = 'Set the regression test\n')
    parser.add_argument('-d',
                        metavar = 'dumpdir',
                        dest = 'dumpdir',
                        type = str,
                        default = './TestFiles',
                        help = 'Set the output dump folder\n')
    parser.add_argument('-p',
                        metavar = 'platform',
                        dest = 'platform',
                        type = str,
                        default = "QEMU-ARM",
                        help = 'Choose the target Platform\n')
    args = parser.parse_args()

    _DEEPLOYSTATEDIR = os.path.join("./TEST_STATE_EQUALITY_DeeployState", args.platform, args.dir)

    onnx_graph = onnx.load_model(f'./{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    inputTypes = {}
    inputOffsets = {}

    inputs = np.load(f'./{args.dir}/inputs.npz')
    tensors = graph.tensors()

    # Load as int64 and infer types later
    test_inputs = [inputs[x].reshape(-1).astype(np.int64) for x in inputs.files]

    platform, signProp = mapPlatform(args.platform)

    for index, num in enumerate(test_inputs):
        _type, offset = inferInputType(num, signProp)[0]
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset

    deployer = mapDeployer(platform, graph, inputTypes, deeployStateDir = _DEEPLOYSTATEDIR, inputOffsets = inputOffsets)

    # Instantiate Classes Requried for Memory Level Annotation Extension
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 1024000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 512000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = 128000)

    memoryHierarchy = MemoryHierarchy([L3, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel("L3")

    deployer.Platform = setupMemoryPlatform(deployer.Platform, memoryHierarchy, defaultTargetMemoryLevel = L1)
    deployer = MemoryDeployerWrapper(deployer)

    deployer.generateFunction()

    ctxt_post_binding_imported = NetworkContext.importNetworkContext(_DEEPLOYSTATEDIR, _backendPostBindingFilename)
    ctxt_pre_lowering_imported = NetworkContext.importNetworkContext(_DEEPLOYSTATEDIR, _middlewarePreLoweringFilename)

    memoryHierarchy = deployer.Platform.memoryHierarchy
    defaultMemoryLevel = deployer.Platform.memoryHierarchy.getDefaultMemoryLevel()

    for bufferName, buffer in ctxt_post_binding_imported.globalObjects.items():
        if isinstance(buffer, VariableBuffer) and not isinstance(buffer, StructBuffer):
            assert buffer._memoryLevel == defaultMemoryLevel.name, f"Tensor {bufferName} in global scope is not annotated with the default memory level"

    for bufferName, buffer in ctxt_post_binding_imported.localObjects.items():
        if isinstance(buffer, VariableBuffer) and not isinstance(buffer, StructBuffer):
            assert buffer._memoryLevel == defaultMemoryLevel.name, f"Tensor {bufferName} in local scope is not annotated with the default memory level"

    assert not ctxt_pre_lowering_imported == deployer.ctxt, "Contexts are not supposed to be equal but are, test failed!"
    assert ctxt_post_binding_imported == deployer.ctxt, "Contexts are supposed to be equal but are not, test failed!"

    # Test if the equality fails correctly if we add a new buffer to the context
    dummyBuffer = VariableBuffer('dummyBuffer')
    alteredCtxt1 = copy.deepcopy(deployer.ctxt)
    alteredCtxt1.globalObjects[dummyBuffer.name] = dummyBuffer
    assert not ctxt_post_binding_imported == alteredCtxt1, "Contexts are not supposed to be equal but are, test failed!"

    # Test if the equality fails correctly if we modify a buffer of the context
    alteredCtxt2 = copy.deepcopy(deployer.ctxt)
    bufferName = list(alteredCtxt2.globalObjects.keys())[0]
    alteredCtxt2.globalObjects[bufferName].name = "meme"
    assert not ctxt_post_binding_imported == alteredCtxt2, "Contexts are not supposed to be equal but are, test failed!"

    print("Contexts equality test passed!")
