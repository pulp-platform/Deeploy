# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.platformMapping import mapDeployer, mapPlatform
from testUtils.testRunner import TestGeneratorArgumentParser, getPaths
from testUtils.typeMapping import inferTypeAndOffset

from Deeploy.CommonExtensions.CodeTransformationPasses.PrintInputs import PrintInputGeneration, PrintOutputGeneration
from Deeploy.MemoryLevelExtension.CodeTransformationPasses.PrintInputs import MemoryAwarePrintInputGeneration, \
    MemoryAwarePrintOutputGeneration
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform

if __name__ == "__main__":

    parser = TestGeneratorArgumentParser(
        description = "Deeploy Code Generation Utility with Print Input/Output Code Transformation")
    args = parser.parse_args()

    inputTypes = {}
    inputOffsets = {}

    _GENDIRROOT = f'TEST_{args.platform.upper()}'
    _GENDIR, _TESTDIR, _TESTNAME = getPaths(args.dir, _GENDIRROOT)

    print("GENDIR    : ", _GENDIR)
    print("TESTDIR   : ", _TESTDIR)
    print("TESTNAME  : ", _TESTNAME)

    _DEEPLOYSTATEDIR = os.path.join(_GENDIR, "TEST_PRINTINPUTOUTPUT", "deeployStates")

    inputs = np.load(f'{args.dir}/inputs.npz')
    outputs = np.load(f'{args.dir}/outputs.npz')
    onnx_graph = onnx.load_model(f'{_TESTDIR}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 1024000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 512000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = 128000)

    memoryHierarchy = MemoryHierarchy([L3, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel("L3")
    defaultTargetMemoryLevel = L1

    platform, signProp = mapPlatform(args.platform)

    for engine in platform.engines:
        for mapping in engine.Mapping:
            for map in engine.Mapping[mapping].maps:
                for binding in map.bindings:
                    if isinstance(platform, MemoryPlatform):
                        binding.codeTransformer.passes += [
                            MemoryAwarePrintInputGeneration(defaultTargetMemoryLevel),
                            MemoryAwarePrintOutputGeneration(defaultTargetMemoryLevel),
                        ]
                    else:
                        binding.codeTransformer.passes += [
                            PrintInputGeneration(),
                            PrintOutputGeneration(),
                        ]

    test_inputs = [inputs[x].reshape(-1).astype(np.float64) for x in inputs.files]
    test_outputs = [outputs[x].reshape(-1).astype(np.float64) for x in outputs.files]
    for index, num in enumerate(test_inputs):
        _type, offset = inferTypeAndOffset(num, signProp)
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset

    deployer = mapDeployer(platform, graph, inputTypes, deeployStateDir = _DEEPLOYSTATEDIR, inputOffsets = inputOffsets)

    deployer.generateFunction()

    print("Print Input/Output Test passed.")
