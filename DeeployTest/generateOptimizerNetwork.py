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

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.codeGenerate import generateOptimizerTestNetwork
from testUtils.platformMapping import mapDeployer, mapPlatform
from testUtils.testRunner import TestGeneratorArgumentParser

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t
from Deeploy.DeeployTypes import _NoVerbosity
from Deeploy.Logging import DEFAULT_LOGGER as log
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
                           name="DeeployOptimizerNetwork",
                           deeployStateDir=_DEEPLOYSTATEDIR,
                           inputOffsets=inputOffsets)

    verbosityCfg = _NoVerbosity
    _ = deployer.prepare(verbosityCfg)

    # 5. Generate OptimizerNetwork.c / OptimizerNetwork.h
    os.makedirs(args.dumpdir, exist_ok=True)
    generateOptimizerTestNetwork(deployer, args.dumpdir, verbosityCfg)

    log.info(f"Optimizer network code generated in: {args.dumpdir}")
    print(f"[OptimizerNetwork] Generated OptimizerNetwork.c/h in {args.dumpdir}")


if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(description="Deeploy Optimizer Network Code Generation.")
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cluster cores. Default: 1.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (informational only; embedded in optimizer ONNX attributes). Default: 0.001.",
    )
    parser.add_argument('--shouldFail', action='store_true')
    parser.set_defaults(shouldFail=False)

    args = parser.parse_args()

    try:
        generateOptimizerNetwork(args)
    except Exception as e:
        if args.shouldFail:
            print("\033[92mOptimizer network generation ended, failed as expected!\033[0m")
            sys.exit(0)
        else:
            raise e

    if args.shouldFail:
        raise RuntimeError("Expected to fail!")
