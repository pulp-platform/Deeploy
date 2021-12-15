# ----------------------------------------------------------------------
#
# File: testSlice_PULP.py
#
# Last edited: 15.06.2023
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

import argparse
import os
import subprocess

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.codeGenerate import generateTestInputsHeader, generateTestNetworkHeader, \
    generateTestNetworkImplementation, generateTestOutputsHeader
from testUtils.platformMapping import mapDeployer, setupMemoryPlatform
from testUtils.testRunner import escapeAnsi
from testUtils.typeMapping import inferInputType

from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.Targets.PULPOpen.Platform import PULPPlatform
from Deeploy.Targets.PULPOpen.Templates.AllocateTemplate import pulpL1AllocateTemplate
from Deeploy.Targets.PULPOpen.Templates.FreeTemplate import pulpL1FreeTemplate

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Test Utility for the Slice Operation.")
    parser.add_argument('--toolchain',
                        metavar = 'toolchain',
                        dest = 'toolchain',
                        type = str,
                        default = "LLVM",
                        help = 'Pick compiler toolchain')
    parser.add_argument('--toolchain_install_dir',
                        metavar = 'toolchain_install_dir',
                        dest = 'toolchain_install_dir',
                        type = str,
                        default = os.environ.get('LLVM_INSTALL_DIR'),
                        help = 'Pick compiler install dir')
    args = parser.parse_args()
    _TOOLCHAIN_DIR = os.path.normpath(args.toolchain_install_dir)

    signProp = False

    onnx_graph = onnx.load_model('./Tests/testSlice/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    inputs = np.load('./Tests/testSlice/inputs.npz')
    outputs = np.load(f'./Tests/testSlice/outputs.npz')
    tensors = graph.tensors()

    # Load as int64 and infer types later
    test_inputs = [inputs[x].reshape(-1).astype(np.int64) for x in inputs.files]
    test_outputs = [outputs[x].reshape(-1).astype(np.int64) for x in outputs.files]

    inputTypes = {}
    inputOffsets = {}

    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 1024000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 512000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = 128000)

    memoryHierarchy = MemoryHierarchy([L3, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel("L2")

    platform = PULPPlatform()

    for index, num in enumerate(test_inputs):
        _type, offset = inferInputType(num, signProp)[0]
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset

    deployer = mapDeployer(platform, graph, inputTypes, inputOffsets = inputOffsets)
    # Make the platform memory-level aware
    deployer.Platform = setupMemoryPlatform(deployer.Platform, memoryHierarchy, defaultTargetMemoryLevel = L1)
    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer)

    # Go through flow
    deployer.frontEnd()
    deployer.parse(deployer.default_channels_first)

    deployer.ctxt.lookup('onnx::Slice_5')._memoryLevel = "L1"
    deployer.ctxt.lookup('onnx::Slice_5').allocTemplate = pulpL1AllocateTemplate
    deployer.ctxt.lookup('onnx::Slice_5').deallocTemplate = pulpL1FreeTemplate

    deployer.midEnd()

    deployer.codeTransform()
    deployer.prepared = True
    deployer.generateInferenceCode()

    # Create input and output vectors
    os.makedirs('TEST_SIRACUSA/Tests/testSlice', exist_ok = True)

    testInputStr = generateTestInputsHeader(deployer, test_inputs, inputTypes, inputOffsets)
    f = open('TEST_SIRACUSA/Tests/testSlice/testinputs.h', "w")
    f.write(testInputStr)
    f.close()

    testOutputStr = generateTestOutputsHeader(deployer, test_outputs, signProp, False)
    f = open('TEST_SIRACUSA/Tests/testSlice/testoutputs.h', "w")
    f.write(testOutputStr)
    f.close()

    # Generate code for Network
    testNetworkHeaderStr = generateTestNetworkHeader(deployer, platform)
    f = open('TEST_SIRACUSA/Tests/testSlice/Network.h', "w")
    f.write(testNetworkHeaderStr)
    f.close()

    testNetworkImplementationStr = generateTestNetworkImplementation(deployer, platform)
    f = open('TEST_SIRACUSA/Tests/testSlice/Network.c', "w")
    f.write(testNetworkImplementationStr)
    f.close()

    clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
    os.system(f'clang-format -i --style="{clang_format}" TEST_SIRACUSA/Tests/testSlice/Network.c')
    os.system(f'clang-format -i --style="{clang_format}" TEST_SIRACUSA/Tests/testSlice/Network.h')
    os.system(f'clang-format -i --style="{clang_format}" TEST_SIRACUSA/Tests/testSlice/testoutputs.h')
    os.system(f'clang-format -i --style="{clang_format}" TEST_SIRACUSA/Tests/testSlice/testinputs.h')

    os.system(
        f"$CMAKE -DTOOLCHAIN={args.toolchain} -DTOOLCHAIN_INSTALL_DIR={_TOOLCHAIN_DIR}  -DTESTNAME=testSlice -DGENERATED_SOURCE=TEST_SIRACUSA/Tests/testSlice -Dplatform=Siracusa -B TEST_SIRACUSA/build -DNUM_CORES=1 .."
    )
    process = subprocess.Popen(["$CMAKE --build TEST_SIRACUSA/build --target gvsoc_testSlice"],
                               stdout = subprocess.PIPE,
                               stderr = subprocess.STDOUT,
                               shell = True,
                               encoding = 'utf-8')
    fileHandle = open('out.txt', 'a')
    fileHandle.write(f"################## Testing Tests/testSlice on SIRACUSA Platform ##################\n")

    result = ""
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            result += output
            fileHandle.write(f"{escapeAnsi(output)}")

    print(result.strip())

    fileHandle.write("")
    fileHandle.close()

    if not "Errors: 0 out of " in result:
        raise RuntimeError(f"Found an error in Tests/testSlice")
