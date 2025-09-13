# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
from testUtils.codeGenerate import generateTestNetwork
from testUtils.dmaUtils import MemcpyLayer, MemcpyParser, MemcpyTileConstraint, MemcpyTypeChecker, generate_graph, \
    memcpyTemplate, prepare_deployer_with_custom_tiling, setup_snitch_deployer
from testUtils.testRunner import TestRunner, TestRunnerArgumentParser
from testUtils.typeMapping import baseTypeFromName, dtypeFromDeeployType

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding, NodeMapper, _NoVerbosity
from Deeploy.Targets.Snitch.Bindings import MemoryAwareFunctionCallClosure, TilingCallClosure
from Deeploy.Targets.Snitch.CodeTransformationPasses import SnitchClusterTiling
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchClusterSynch import SnitchSynchCoresPass
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchCoreFilter import SnitchCoreFilterPass
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchProfileExecutionBlock import SnitchProfileExecutionBlockPass
from Deeploy.Targets.Snitch.DMA.SnitchDma import SnitchDma
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement, \
    TilingVariableReplacementUpdate
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

testRunnerArgumentParser = TestRunnerArgumentParser(tiling_arguments = True)
testRunnerArgumentParser.add_argument('--input-shape',
                                      nargs = '+',
                                      required = True,
                                      dest = 'input_shape',
                                      type = int,
                                      help = "Shape of the copied tensor")
testRunnerArgumentParser.add_argument('--tile-shape',
                                      nargs = '+',
                                      required = True,
                                      dest = 'tile_shape',
                                      type = int,
                                      help = "Shape of the tiles produced in the manual tiling solution")
testRunnerArgumentParser.add_argument('--node-count',
                                      dest = 'node_count',
                                      type = int,
                                      default = 1,
                                      help = "Number of generated memcpy nodes")
testRunnerArgumentParser.add_argument('--type', type = str, default = "uint8_t", help = "Tensor elements datatype")
testRunner = TestRunner('Snitch', 'gvsoc', tiling = True, argument_parser = testRunnerArgumentParser)

inputShape = testRunner._args.input_shape
tileShape = testRunner._args.tile_shape
node_count = testRunner._args.node_count
_type = baseTypeFromName(testRunner._args.type)
dtype = dtypeFromDeeployType(_type)
defaultMemory = "L2"
targetMemory = "L1"

assert len(inputShape) == len(tileShape), \
    f'Input and tile shape should be of the same dimensionality. Received {len(inputShape)}D input shape vs. {len(tileShape)}D tile shape.'
assert all(tileDim <= inDim for inDim, tileDim in zip(inputShape, tileShape)), \
    f'Each tile shape dimension should be smaller then the corresponding input one. Received {tileShape} > {inputShape}'

graph = generate_graph(node_count, inputShape, dtype)
inputTypes = {"input_0": PointerClass(_type)}
_DEEPLOYSTATEDIR = os.path.join(testRunner._dir_gen, "deeployStates")
deployer = setup_snitch_deployer(defaultMemory, targetMemory, graph, inputTypes, testRunner._args.doublebuffer,
                                 _DEEPLOYSTATEDIR)

transformer = CodeTransformation([
    SnitchCoreFilterPass("compute"),
    SnitchProfileExecutionBlockPass(),
    TilingVariableReplacement(targetMemory),
    TilingCallClosure(writeback = False),
    SnitchSynchCoresPass(),
    TilingVariableReplacementUpdate(targetMemory),
    SnitchClusterTiling(defaultMemory, targetMemory, SnitchDma()),
    ArgumentStructGeneration(),
    MemoryManagementGeneration(targetMemory),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    MemoryManagementGeneration(defaultMemory),
    MemoryManagementGeneration(),
])

binding = NodeBinding(MemcpyTypeChecker(), memcpyTemplate, transformer)
tilingReadyBindings = TilingReadyNodeBindings([binding], MemcpyTileConstraint())
memcpyMapper = NodeMapper(MemcpyParser(), tilingReadyBindings)
memcpyMapping = {"Memcpy": MemcpyLayer([memcpyMapper])}
deployer.Platform.engines[0].Mapping.update(memcpyMapping)

prepare_deployer_with_custom_tiling(deployer, defaultMemory, targetMemory, tileShape, testRunner._args.doublebuffer)

if not testRunner._args.skipgen:
    if dtype == np.float32:
        test_inputs = np.random.rand(*inputShape)
    else:
        info = np.iinfo(dtype)
        test_inputs = np.arange(stop = np.prod(inputShape), dtype = dtype).reshape(inputShape)
    test_outputs = test_inputs
    generateTestNetwork(deployer, [test_inputs], [test_outputs], testRunner._dir_gen, _NoVerbosity)

# Deconstructed testRunner.run() with skipped generation because we did the generation already
testRunner.configure_cmake_project()
testRunner.build_binary()
if not testRunner._args.skipsim:
    testRunner.run_simulation()
