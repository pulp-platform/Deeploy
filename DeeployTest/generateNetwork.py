# ----------------------------------------------------------------------
#
# File: generateNetwork.py
#
# Last edited: 08.01.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, ETH Zurich
# - Philip Wiese, ETH Zurich
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

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.codeGenerate import generateTestInputsHeader, generateTestNetworkHeader, \
    generateTestNetworkImplementation, generateTestOutputsHeader
from testUtils.graphDebug import generateDebugConfig
from testUtils.platformMapping import mapDeployer, mapPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.typeMapping import inferInputType

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.DebugPasses import EmulateCMSISRequantPass
from Deeploy.DeeployTypes import _NoVerbosity
from Deeploy.Targets.CortexM.Platform import CMSISPlatform

_TEXT_ALIGN = 30

if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(description = "Deeploy Code Generation Utility.")
    parser.add_argument('--debug',
                        dest = 'debug',
                        action = 'store_true',
                        default = False,
                        help = 'Enable debugging mode\n')
    parser.add_argument('--overwriteRecentState',
                        action = 'store_true',
                        help = 'Copy the recent deeply state to the ./deeployStates folder\n')

    args = parser.parse_args()

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

    platform, signProp = mapPlatform(args.platform)

    for index, num in enumerate(test_inputs):
        # WIESP: Do not infer types and offset of empty arrays
        if np.prod(num.shape) == 0:
            continue
        _type, offset = inferInputType(num, signProp)[0]
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset

    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    deployer = mapDeployer(platform, graph, inputTypes, deeployStateDir = _DEEPLOYSTATEDIR, inputOffsets = inputOffsets)

    if not isinstance(
            platform, CMSISPlatform
    ) and not "simpleCNN" in args.dir and not "testRQMatMul" in args.dir and not "testRQGEMM" in args.dir:
        deployer.loweringOptimizer.passes.insert(0, EmulateCMSISRequantPass())

    # Parse graph and infer output levels and signedness
    _ = deployer.generateFunction(verbose = _NoVerbosity)

    if args.overwriteRecentState:
        os.makedirs(f'./deeployStates/', exist_ok = True)
        os.system(f'cp -r {_DEEPLOYSTATEDIR}/* ./deeployStates/')

    # Create input and output vectors
    os.makedirs(f'{args.dumpdir}', exist_ok = True)

    testInputStr = generateTestInputsHeader(deployer, test_inputs, inputTypes, inputOffsets)
    f = open(f'{args.dumpdir}/testinputs.h', "w")
    f.write(testInputStr)
    f.close()

    testOutputStr = generateTestOutputsHeader(deployer, test_outputs, signProp, verbose = args.verbose)
    f = open(f'{args.dumpdir}/testoutputs.h', "w")
    f.write(testOutputStr)
    f.close()

    # Generate code for Network
    testNetworkHeaderStr = generateTestNetworkHeader(deployer, platform)
    f = open(f'{args.dumpdir}/Network.h', "w")
    f.write(testNetworkHeaderStr)
    f.close()

    testNetworkImplementationStr = generateTestNetworkImplementation(deployer, platform, verbose = args.verbose)
    f = open(f'{args.dumpdir}/Network.c', "w")
    f.write(testNetworkImplementationStr)
    f.close()

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
        print(f"{'Model Parameters: ' :<{_TEXT_ALIGN}} {deployer.getParameterSize()}")
