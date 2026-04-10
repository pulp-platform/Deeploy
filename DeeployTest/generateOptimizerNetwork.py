# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Optimizer network code-generation entry point.

Loads the optimizer ONNX graph (containing Deeploy SGD nodes) and emits
OptimizerNetwork.c / OptimizerNetwork.h into the specified output directory.

The generated code uses the prefix ``DeeployOptNetwork_`` (instead of the
default ``DeeployNetwork_``) so that it can be linked together with the
training network without symbol conflicts.

Usage
-----
    /usr/bin/python generateOptimizerNetwork.py \\
        -t <optimizer_dir>  \\   # directory containing network.onnx
        -d <output_dir>     \\   # where to write OptimizerNetwork.c/h
        -p Siracusa         \\
        --cores 8           \\
        --lr 0.001
"""

import os
import sys
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from testUtils.codeGenerate import build_shared_buffer_maps, generateOptimizerTestNetwork
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.trainingUtils import add_cores_arg, add_memory_level_args, add_optimizer_training_dir_arg, \
    add_should_fail_arg, run_with_shouldfail

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t
from Deeploy.DeeployTypes import _NoVerbosity
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine


def generateOptimizerNetwork(args):
    log.debug("Arguments: %s", args)

    # 1. Load optimizer network.onnx
    onnx_path = f'{args.dir}/network.onnx'
    onnx_model = onnx.load_model(onnx_path)
    graph = gs.import_onnx(onnx_model)

    log.debug(f"Optimizer ONNX inputs: {[i.name for i in onnx_model.graph.input]}")
    log.debug(f"Optimizer ONNX outputs: {[o.name for o in onnx_model.graph.output]}")

    # 2. Platform setup
    platform, signProp = mapPlatform(args.platform)
    log.debug(f"Platform: {platform} (sign: {signProp})")

    clusters = [e for e in platform.engines if isinstance(e, PULPClusterEngine)]
    for cluster in clusters:
        cluster.n_cores = args.cores

    # 3. All optimizer inputs are float32 (weights + grad acc buffers).
    graph_input_names = [inp.name for inp in onnx_model.graph.input]
    inputTypes = {f"input_{i}": PointerClass(float32_t) for i in range(len(graph_input_names))}
    inputOffsets = {f"input_{i}": 0 for i in range(len(graph_input_names))}

    # 4. Create and prepare deployer
    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates_optimizer")

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           name = "DeeployOptimizerNetwork",
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets)

    # Set up memory hierarchy so AnnotateDefaultMemoryLevel assigns the correct
    # memory level to ConstantBuffers (weights).  The optimizer graph is NOT
    # tiled, but it must share the same memory-level view as the training graph
    # so that weights end up in the same physical location (L2 when L3 is the
    # training default, see AnnotateDefaultMemoryLevel).
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 64000000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = args.l2)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = args.l1)
    memoryHierarchy = MemoryHierarchy([L3, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel(args.defaultMemLevel)
    defaultTargetMemoryLevel = memoryHierarchy.memoryLevels[args.defaultMemLevel]

    deployer.Platform = setupMemoryPlatform(deployer.Platform, memoryHierarchy, defaultTargetMemoryLevel)
    deployer = MemoryDeployerWrapper(deployer, [AnnotateDefaultMemoryLevel(memoryHierarchy)])

    verbosityCfg = _NoVerbosity
    _ = deployer.prepare(verbosityCfg)

    # 5. Build shared-buffer maps when the training ONNX is available
    shared_input_map: dict = {}
    shared_output_map: dict = {}
    training_onnx = Path(args.training_dir) / "network.onnx" if args.training_dir else None
    if training_onnx and training_onnx.exists():
        shared_input_map, shared_output_map = build_shared_buffer_maps(str(training_onnx), onnx_model)
        log.debug(f"[SharedBuffers] input map: {shared_input_map}")
        log.debug(f"[SharedBuffers] output map: {shared_output_map}")
        log.info(f"[OptimizerNetwork] Sharing {len(shared_input_map)} inputs and "
                 f"{len(shared_output_map)} outputs with TrainingNetwork")
    else:
        if args.training_dir:
            log.warning(f"[OptimizerNetwork] training_dir set but {training_onnx} not found — "
                        "generating standalone OptimizerNetwork (no buffer sharing)")

    # 6. Generate OptimizerNetwork.c / OptimizerNetwork.h
    os.makedirs(args.dumpdir, exist_ok = True)
    generateOptimizerTestNetwork(deployer, args.dumpdir, verbosityCfg, shared_input_map, shared_output_map)

    log.info(f"Optimizer network code generated in: {args.dumpdir}")
    print(f"[OptimizerNetwork] Generated OptimizerNetwork.c/h in {args.dumpdir}")


if __name__ == '__main__':
    parser = TestGeneratorArgumentParser(description = "Deeploy Optimizer Network Code Generation.")
    add_cores_arg(parser)
    parser.add_argument(
        "--lr",
        type = float,
        default = 0.001,
        help = "Learning rate (informational only; embedded in optimizer ONNX attributes). Default: 0.001.",
    )
    add_memory_level_args(parser)
    add_optimizer_training_dir_arg(parser)
    add_should_fail_arg(parser)
    args = parser.parse_args()
    run_with_shouldfail(generateOptimizerNetwork, args, "Optimizer network generation")
