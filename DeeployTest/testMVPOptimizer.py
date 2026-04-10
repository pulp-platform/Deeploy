# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiled optimizer network code-generation entry point.

Loads the optimizer ONNX graph (containing Deeploy SGD nodes) and emits
OptimizerNetwork.c / OptimizerNetwork.h into the specified output directory,
using the SB-Tiler to tile SGD kernels through L1.

The generated code uses the prefix ``DeeployOptNetwork_`` (instead of the
default ``DeeployNetwork_``) so that it can be linked together with the
training network without symbol conflicts.

Usage
-----
    /usr/bin/python testMVPOptimizer.py \\
        -t <optimizer_dir>  \\   # directory containing network.onnx
        -d <output_dir>     \\   # where to write OptimizerNetwork.c/h
        -p Siracusa         \\
        --cores 8           \\
        --l1 64000          \\
        --l2 1024000        \\
        --defaultMemLevel L2
"""

import hashlib
import os
import sys
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from testUtils.codeGenerateTraining import build_shared_buffer_maps, generateOptimizerTestNetwork
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.tilingUtils import TrainingSBTiler
from testUtils.trainingUtils import _mockScheduler, add_optimizer_training_dir_arg

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t
from Deeploy.DeeployTypes import CodeGenVerbosity, _NoVerbosity
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, \
    AnnotateIOMemoryLevel
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine
from Deeploy.TilingExtension.TilerExtension import TilerDeployerWrapper


def generateTiledOptimizerNetwork(args) -> None:
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

    # 4. Create deployer with _mockScheduler (required for TilerDeployerWrapper).
    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates_optimizer")

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           name = "DeeployOptimizerNetwork",
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets,
                           scheduler = _mockScheduler)

    # 5. Set up memory hierarchy.
    #    Tiles execute in L1; optimizer I/O (weights, grads) live in L2 (or L3).
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 64_000_000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = args.l2)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = args.l1)
    memoryHierarchy = MemoryHierarchy([L3, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel(args.defaultMemLevel)

    defaultTargetMemLevel = L1
    defaultIoMemLevel = memoryHierarchy.memoryLevels[args.defaultMemLevel]

    # 6. Wrap with memory-level annotation.
    deployer.Platform = setupMemoryPlatform(deployer.Platform, memoryHierarchy, defaultTargetMemLevel)
    deployer = MemoryDeployerWrapper(deployer, [
        AnnotateIOMemoryLevel(defaultIoMemLevel.name),
        AnnotateDefaultMemoryLevel(memoryHierarchy),
    ])

    # 7. Wrap with SBTiler (single-buffering; optimizer is forward-only, no lifetime extension needed).
    unique_params = f"{args.dumpdir}_L1{args.l1}_L2{args.l2}_{args.defaultMemLevel}_optimizer"
    testIdentifier = hashlib.md5(unique_params.encode()).hexdigest()[:16]

    # TrainingSBTiler extends all input buffer lifetimes to the end of the
    # schedule (via TrainingMemoryScheduler).  This prevents the allocator from
    # reusing the space of a consumed input (e.g. fc1 weight) for a later
    # output (e.g. fc2 updated weight), which would corrupt the weight buffer.
    deployer = TilerDeployerWrapper(deployer, TrainingSBTiler, testName = testIdentifier, workDir = args.dumpdir)
    deployer.tiler.visualizeMemoryAlloc = args.plotMemAlloc
    deployer.tiler.memoryAllocStrategy = args.memAllocStrategy
    deployer.tiler.searchStrategy = args.searchStrategy

    # 8. Prepare deployer.
    verbosityCfg = _NoVerbosity
    if args.profileTiling:
        verbosityCfg = CodeGenVerbosity(tilingProfiling = True)
    _ = deployer.prepare(verbosityCfg)

    # 9. Build shared-buffer maps when the training ONNX is available
    shared_input_map: dict = {}
    shared_output_map: dict = {}
    training_onnx = Path(args.training_dir) / "network.onnx" if args.training_dir else None
    if training_onnx and training_onnx.exists():
        shared_input_map, shared_output_map = build_shared_buffer_maps(str(training_onnx), onnx_model)
        log.debug(f"[SharedBuffers] input map: {shared_input_map}")
        log.debug(f"[SharedBuffers] output map: {shared_output_map}")
        log.info(f"[TiledOptimizerNetwork] Sharing {len(shared_input_map)} inputs and "
                 f"{len(shared_output_map)} outputs with TrainingNetwork")
    else:
        if args.training_dir:
            log.warning(f"[TiledOptimizerNetwork] training_dir set but {training_onnx} not found — "
                        "generating standalone OptimizerNetwork (no buffer sharing)")

    # 10. Generate OptimizerNetwork.c / OptimizerNetwork.h
    os.makedirs(args.dumpdir, exist_ok = True)
    generateOptimizerTestNetwork(deployer, args.dumpdir, verbosityCfg, shared_input_map, shared_output_map)

    log.info(f"Tiled optimizer network code generated in: {args.dumpdir}")
    print(f"[TiledOptimizerNetwork] Generated OptimizerNetwork.c/h in {args.dumpdir}")


if __name__ == '__main__':
    parser = TestGeneratorArgumentParser(description = "Deeploy Tiled Optimizer Network Code Generation.")
    parser.add_argument("--cores", type = int, default = 1, help = "Number of cluster cores. Default: 1.")
    parser.add_argument(
        "--lr",
        type = float,
        default = 0.001,
        help = "Learning rate (informational only; embedded in optimizer ONNX attributes). Default: 0.001.",
    )
    parser.add_argument("--l1", type = int, default = 64_000, help = "L1 size in bytes. Default: 64000.")
    parser.add_argument("--l2", type = int, default = 1_024_000, help = "L2 size in bytes. Default: 1024000.")
    parser.add_argument("--defaultMemLevel",
                        type = str,
                        default = "L2",
                        help = "Default memory level for IO buffers. Default: L2.")
    parser.add_argument("--memAllocStrategy",
                        type = str,
                        default = "MiniMalloc",
                        help = "Memory allocation strategy. Default: MiniMalloc.")
    parser.add_argument("--searchStrategy",
                        type = str,
                        default = "random-max",
                        help = "CP solver search strategy. Default: random-max.")
    parser.add_argument("--plotMemAlloc",
                        action = "store_true",
                        help = "Save memory allocation plots in the deeployStates folder.")
    parser.add_argument("--profileTiling",
                        action = "store_true",
                        help = "Enable tiling profiling (inserts cycle counters around each tiled kernel).")
    add_optimizer_training_dir_arg(parser)
    parser.add_argument("--shouldFail", action = "store_true")
    parser.set_defaults(shouldFail = False)
    args = parser.parse_args()

    try:
        generateTiledOptimizerNetwork(args)
    except Exception:
        if args.shouldFail:
            print("\033[92mTiled optimizer network generation ended, failed as expected!\033[0m")
            sys.exit(0)
        raise
    if args.shouldFail:
        raise RuntimeError("Expected to fail!")
