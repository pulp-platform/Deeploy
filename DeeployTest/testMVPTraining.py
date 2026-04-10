# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import sys
from typing import List

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.codeGenerate import generateTrainingTestNetwork
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.tilingUtils import TrainingSBTiler
from testUtils.typeMapping import inferTypeAndOffset

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t, uint8_t
from Deeploy.DeeployTypes import CodeGenVerbosity, _NoVerbosity
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, \
    AnnotateIOMemoryLevel
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine
from Deeploy.TilingExtension.TilerExtension import TilerDeployerWrapper

_GRAD_ACC = "_grad.accumulation.buffer"

# ---------------------------------------------------------------------------
# Helpers copied from generateTrainingNetwork.py
# ---------------------------------------------------------------------------


def _load_reference_losses(train_dir: str) -> list:
    """Load reference loss values from outputs.npz."""
    outputs_path = os.path.join(train_dir, "outputs.npz")
    if not os.path.exists(outputs_path):
        log.warning(f"outputs.npz not found at {outputs_path} — loss comparison skipped")
        return None
    try:
        outputs = np.load(outputs_path)
    except Exception as e:
        log.warning(f"Failed to load outputs.npz: {e} — loss comparison skipped")
        return None
    for key in outputs.files:
        if 'loss' in key.lower():
            vals = [float(v) for v in np.array(outputs[key]).flatten().tolist()]
            log.info(f"Reference losses loaded from outputs.npz['{key}']: {vals}")
            return vals
    log.warning("No 'loss' key found in outputs.npz — loss comparison skipped")
    return None


def _infer_num_data_inputs(inputs_path: str) -> int:
    inputs = np.load(inputs_path)
    base_keys = sorted(k for k in inputs.files if not k.startswith('mb') and not k.startswith('meta_'))
    count = sum(1 for k in base_keys if f'mb1_{k}' in inputs.files)
    if count == 0:
        raise ValueError("Cannot auto-detect num_data_inputs: inputs.npz has only one mini-batch "
                         "(no mb1_arr_* entries found). Please pass --num-data-inputs explicitly.")
    return count


def _infer_total_mb(inputs_path: str) -> int:
    inputs = np.load(inputs_path)
    if "meta_n_batches" in inputs.files:
        return int(inputs["meta_n_batches"].flat[0])
    mb_indices = set()
    for key in inputs.files:
        if key.startswith('mb'):
            try:
                idx = int(key.split('_')[0][2:])
                mb_indices.add(idx)
            except ValueError:
                pass
    return 1 + len(mb_indices)


def _infer_data_size(inputs_path: str) -> int:
    inputs = np.load(inputs_path)
    if "meta_data_size" in inputs.files:
        return int(inputs["meta_data_size"].flat[0])
    return _infer_total_mb(inputs_path)


def _infer_n_accum(inputs_path: str) -> int:
    inputs = np.load(inputs_path)
    if "meta_n_accum" in inputs.files:
        return int(inputs["meta_n_accum"].flat[0])
    return 1


# ---------------------------------------------------------------------------
# Mock scheduler (same as testMVP.py)
# ---------------------------------------------------------------------------


def _mockScheduler(graph: gs.Graph) -> List[List[gs.Node]]:
    """Wrap every node in a singleton list for the Tiler pattern interface."""
    return [[node] for node in graph.nodes]


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generateTiledTrainingNetwork(args) -> None:
    log.debug("Arguments: %s", args)

    # 1. Load network.onnx (training graph with forward + backward ops).
    onnx_graph = onnx.load_model(f'{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    # 1a. Strip UNDEFINED-typed unused optional outputs (e.g. MaxPool mask indices).
    _stripped = False
    for node in graph.nodes:
        filtered = [out for out in node.outputs if not (out.dtype == 0 and len(out.outputs) == 0)]
        if len(filtered) < len(node.outputs):
            node.outputs = filtered
            _stripped = True
    if _stripped:
        graph.cleanup()
        log.debug("Stripped UNDEFINED-typed unused optional outputs from graph nodes")

    # 2. Load inputs.npz.
    inputs_path = f'{args.dir}/inputs.npz'
    inputs = np.load(inputs_path)

    # 3. Platform setup.
    platform, signProp = mapPlatform(args.platform)
    log.debug(f"Platform: {platform} (sign: {signProp})")

    clusters = [engine for engine in platform.engines if isinstance(engine, PULPClusterEngine)]
    for cluster in clusters:
        cluster.n_cores = args.cores

    # 4. Identify grad acc buf positions in the ONNX graph.
    graph_input_names = [inp.name for inp in onnx_graph.graph.input]
    grad_acc_set = {i for i, n in enumerate(graph_input_names) if _GRAD_ACC in n}
    non_grad_indices = [i for i in range(len(graph_input_names)) if i not in grad_acc_set]

    base_keys = sorted(k for k in inputs.files if not k.startswith('mb') and not k.startswith('meta_'))
    npz_base = [inputs[k] for k in base_keys]

    if len(npz_base) != len(non_grad_indices):
        raise ValueError(f"inputs.npz has {len(npz_base)} base entries but network.onnx has "
                         f"{len(non_grad_indices)} non-grad-buf inputs. "
                         f"Re-generate inputs.npz with the updated exporter.")

    # 5. Build inputTypes / inputOffsets for ALL graph input positions.
    inputTypes = {}
    inputOffsets = {}

    npz_idx = 0
    for graph_idx, name in enumerate(graph_input_names):
        if graph_idx in grad_acc_set:
            inputTypes[f"input_{graph_idx}"] = PointerClass(float32_t)
            inputOffsets[f"input_{graph_idx}"] = 0
        else:
            arr = npz_base[npz_idx]
            npz_idx += 1
            if arr.dtype == bool or arr.dtype == np.bool_:
                inputTypes[f"input_{graph_idx}"] = PointerClass(uint8_t)
                inputOffsets[f"input_{graph_idx}"] = 0
            elif arr.dtype in (np.float32, np.float64):
                inputTypes[f"input_{graph_idx}"] = PointerClass(float32_t)
                inputOffsets[f"input_{graph_idx}"] = 0
            elif np.prod(arr.shape) == 0:
                pass
            else:
                values = arr.reshape(-1).astype(np.float32)
                _type, offset = inferTypeAndOffset(values, signProp = False)
                inputTypes[f"input_{graph_idx}"] = _type
                inputOffsets[f"input_{graph_idx}"] = offset

    # 6. Create deployer with _mockScheduler (required for TilerDeployerWrapper).
    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           name = "DeeployTrainingNetwork",
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets,
                           scheduler = _mockScheduler)

    # 7. Set up memory hierarchy.
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 64_000_000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = args.l2)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = args.l1)
    memoryHierarchy = MemoryHierarchy([L3, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel(args.defaultMemLevel)

    defaultTargetMemLevel = L1
    defaultIoMemLevel = memoryHierarchy.memoryLevels[args.defaultMemLevel]

    # 8. Wrap with memory-level annotation.
    deployer.Platform = setupMemoryPlatform(deployer.Platform, memoryHierarchy, defaultTargetMemLevel)

    deployer = MemoryDeployerWrapper(deployer, [
        AnnotateIOMemoryLevel(defaultIoMemLevel.name),
        AnnotateDefaultMemoryLevel(memoryHierarchy),
    ])

    # 9. Wrap with tiler (TrainingSBTiler: SB strategy + extended input lifetimes for backward pass).
    unique_params = f"{args.dumpdir}_L1{args.l1}_L2{args.l2}_{args.defaultMemLevel}"
    testIdentifier = hashlib.md5(unique_params.encode()).hexdigest()[:16]

    deployer = TilerDeployerWrapper(deployer, TrainingSBTiler, testName = testIdentifier, workDir = args.dumpdir)
    deployer.tiler.visualizeMemoryAlloc = args.plotMemAlloc
    deployer.tiler.memoryAllocStrategy = args.memAllocStrategy
    deployer.tiler.searchStrategy = args.searchStrategy

    # 10. Prepare deployer.
    verbosityCfg = _NoVerbosity
    if args.profileTiling:
        verbosityCfg = CodeGenVerbosity(tilingProfiling = True)
    _ = deployer.prepare(verbosityCfg)

    # 11. Resolve num_data_inputs, n_steps, n_accum.
    num_data = args.num_data_inputs
    if num_data is None:
        num_data = _infer_num_data_inputs(inputs_path)
        log.info(f"Auto-detected num_data_inputs={num_data} from inputs.npz")

    n_steps = args.n_steps
    n_accum = args.n_accum
    if n_steps is None or n_accum is None:
        total_mb = _infer_total_mb(inputs_path)
        log.info(f"Auto-detected total_mb={total_mb} from inputs.npz")
        if n_steps is None and n_accum is None:
            n_accum = _infer_n_accum(inputs_path)
            n_steps = max(1, total_mb // n_accum)
        elif n_steps is None:
            n_steps = max(1, total_mb // n_accum)
        else:
            n_accum = max(1, total_mb // n_steps)

    log.info(f"Training config: n_steps={n_steps} n_accum={n_accum} num_data_inputs={num_data}")

    # 12. Build unique_mb_data from npz.
    total_mb = n_steps * n_accum
    data_size = _infer_data_size(inputs_path)
    log.info(f"Data cycling: data_size={data_size}, total_mb={total_mb}")
    mb0_data = list(npz_base[:num_data])

    unique_mb_data = []
    for mb in range(data_size):
        if mb == 0:
            unique_mb_data.append(mb0_data)
        else:
            mb_row = []
            for buf_idx in range(num_data):
                key = f"mb{mb}_arr_{buf_idx:04d}"
                mb_row.append(inputs[key] if key in inputs else mb0_data[buf_idx])
            unique_mb_data.append(mb_row)

    # Grad acc buf info for testinputs.h.
    if grad_acc_set:
        sorted_grad = sorted(grad_acc_set)
        grad_buf_start_idx = sorted_grad[0]
    else:
        grad_buf_start_idx = -1
    num_grad_inputs = len(grad_acc_set)

    if grad_buf_start_idx > num_data:
        init_weights = list(npz_base[num_data:grad_buf_start_idx])
    else:
        init_weights = []

    # 13. Load reference losses.
    reference_losses = _load_reference_losses(args.dir)

    # 14. Generate output files.
    os.makedirs(args.dumpdir, exist_ok = True)

    generateTrainingTestNetwork(deployer,
                                unique_mb_data,
                                args.dumpdir,
                                verbosityCfg,
                                n_steps = n_steps,
                                n_accum = n_accum,
                                num_data_inputs = num_data,
                                grad_buf_start_idx = grad_buf_start_idx,
                                num_grad_inputs = num_grad_inputs,
                                learning_rate = args.learning_rate,
                                reference_losses = reference_losses,
                                init_weights = init_weights,
                                data_size = data_size,
                                tolerance_abs = args.tolerance_abs)

    # 15. Write resolved config for execution.py to pick up.
    meta = {
        "n_train_steps": n_steps,
        "n_accum_steps": n_accum,
        "training_num_data_inputs": num_data,
    }
    meta_path = os.path.join(args.dumpdir, "training_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent = 2)
    log.info(f"Training meta written to {meta_path}: {meta}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(description = "Deeploy Tiled Training Code Generation Utility.")

    # Training params (same as generateTrainingNetwork.py)
    parser.add_argument(
        "--cores",
        type = int,
        default = 1,
        help = "Number of cores on which the network is run. Default: 1.",
    )
    parser.add_argument(
        "--num-data-inputs",
        type = int,
        dest = "num_data_inputs",
        default = None,
        help = "Number of DATA inputs that change per mini-batch. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--n-steps",
        type = int,
        dest = "n_steps",
        default = None,
        help = "N_TRAIN_STEPS: number of gradient-accumulation update steps.",
    )
    parser.add_argument(
        "--n-accum",
        type = int,
        dest = "n_accum",
        default = None,
        help = "N_ACCUM_STEPS: number of mini-batches per update step.",
    )
    parser.add_argument(
        "--learning-rate",
        type = float,
        dest = "learning_rate",
        default = 0.001,
        help = "SGD learning rate emitted as TRAINING_LEARNING_RATE in testinputs.h. Default: 0.001.",
    )

    # Tiling params (same as testMVP.py)
    parser.add_argument(
        '--l1',
        type = int,
        dest = 'l1',
        default = 64_000,
        help = 'Set L1 size in bytes. Default: 64000.',
    )
    parser.add_argument(
        '--l2',
        type = int,
        dest = 'l2',
        default = 1_024_000,
        help = 'Set L2 size in bytes. Default: 1024000.',
    )
    parser.add_argument(
        '--defaultMemLevel',
        type = str,
        dest = 'defaultMemLevel',
        default = "L2",
        help = 'Default memory level for IO buffers. Default: L2.',
    )
    parser.add_argument(
        '--memAllocStrategy',
        type = str,
        dest = 'memAllocStrategy',
        default = "MiniMalloc",
        help = 'Memory allocation strategy. Default: MiniMalloc.',
    )
    parser.add_argument(
        '--searchStrategy',
        type = str,
        dest = 'searchStrategy',
        default = "random-max",
        help = 'CP solver search strategy. Default: random-max.',
    )
    parser.add_argument(
        '--plotMemAlloc',
        action = 'store_true',
        help = 'Save memory allocation plots in the deeployStates folder.',
    )
    parser.add_argument(
        '--profileTiling',
        action = 'store_true',
        help = 'Enable tiling profiling (inserts cycle counters around each tiled kernel).',
    )
    parser.add_argument(
        '--tolerance',
        type = float,
        dest = 'tolerance_abs',
        default = 1e-3,
        help = 'Absolute loss tolerance emitted as TRAINING_TOLERANCE_ABS in testoutputs.h. Default: 1e-3.',
    )
    parser.add_argument('--shouldFail', action = 'store_true')
    parser.set_defaults(shouldFail = False)

    args = parser.parse_args()

    try:
        generateTiledTrainingNetwork(args)
    except Exception as e:
        if args.shouldFail:
            print("\033[92mTiled training network generation ended, failed as expected!\033[0m")
            sys.exit(0)
        else:
            raise e

    if args.shouldFail:
        raise RuntimeError("Expected to fail!")
