# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.codeGenerateTraining import generateTrainingTestNetwork
from testUtils.platformMapping import mapDeployer, mapPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.trainingUtils import _GRAD_ACC, _infer_data_size, _infer_n_accum, _infer_num_data_inputs, \
    _infer_total_mb, _load_reference_losses, add_training_inference_args
from testUtils.typeMapping import inferTypeAndOffset

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t, uint8_t
from Deeploy.DeeployTypes import _NoVerbosity
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine


def generateTrainingNetwork(args):
    log.debug("Arguments: %s", args)

    # 1. Load network.onnx (training graph)
    onnx_graph = onnx.load_model(f'{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    # 1a. Handle UNDEFINED-typed outputs in training ONNX graphs.
    # Backward pass ONNX often doesn't propagate types for gradient outputs.
    # (i) Strip UNDEFINED-typed outputs that have no consumers.
    # (ii) Patch UNDEFINED-typed outputs WITH consumers to float32 (training default).
    _stripped = False
    _patched = False
    for node in graph.nodes:
        filtered = [out for out in node.outputs if not (out.dtype == 0 and len(out.outputs) == 0)]
        if len(filtered) < len(node.outputs):
            node.outputs = filtered
            _stripped = True
        for out in node.outputs:
            if out.dtype == 0 and len(out.outputs) > 0:
                out.dtype = np.dtype(np.float32)
                _patched = True
    if _stripped:
        graph.cleanup()
        log.debug("Stripped UNDEFINED-typed unused optional outputs from graph nodes")
    if _patched:
        log.debug("Patched UNDEFINED-typed outputs with consumers to float32")

    # 2. Load inputs.npz (new format: no grad acc buf entries)
    inputs_path = f'{args.dir}/inputs.npz'
    inputs = np.load(inputs_path)

    # 3. Platform setup
    platform, signProp = mapPlatform(args.platform)

    log.debug(f"Platform: {platform} (sign: {signProp})")

    # Set cores on cluster engines (same pattern as generateNetwork.py)
    clusters = [engine for engine in platform.engines if isinstance(engine, PULPClusterEngine)]
    for cluster in clusters:
        cluster.n_cores = args.cores

    # 4. Identify grad acc buf positions in the ONNX graph.
    graph_input_names = [inp.name for inp in onnx_graph.graph.input]
    grad_acc_set = {i for i, n in enumerate(graph_input_names) if _GRAD_ACC in n}
    non_grad_indices = [i for i in range(len(graph_input_names)) if i not in grad_acc_set]

    # Base npz arrays: keys that are neither per-mb entries (mb*) nor metadata (meta_*)
    base_keys = sorted(k for k in inputs.files if not k.startswith('mb') and not k.startswith('meta_'))
    npz_base = [inputs[k] for k in base_keys]

    if len(npz_base) != len(non_grad_indices):
        raise ValueError(f"inputs.npz has {len(npz_base)} base entries but network.onnx has "
                         f"{len(non_grad_indices)} non-grad-buf inputs. "
                         f"Re-generate inputs.npz with the updated exporter.")

    # Build inputTypes / inputOffsets for ALL graph input positions.
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
                # Float32 training parameters always stay float32.
                # inferTypeAndOffset would misclassify integer-valued floats
                # (e.g. LayerNorm gamma=1.0 / beta=0.0) as int8_t.
                inputTypes[f"input_{graph_idx}"] = PointerClass(float32_t)
                inputOffsets[f"input_{graph_idx}"] = 0
            elif np.prod(arr.shape) == 0:
                pass
            else:
                values = arr.reshape(-1).astype(np.float32)
                _type, offset = inferTypeAndOffset(values, signProp = False)
                inputTypes[f"input_{graph_idx}"] = _type
                inputOffsets[f"input_{graph_idx}"] = offset

    # 5. Create deployer
    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           name = "DeeployTrainingNetwork",
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets)

    log.debug(f"Deployer: {deployer}")

    # 6. Prepare deployer
    verbosityCfg = _NoVerbosity

    _ = deployer.prepare(verbosityCfg)

    # 7. Resolve num_data_inputs, n_steps, n_accum (auto-detect when not given).

    # num_data_inputs: detect from npz mb1 variants if not specified
    num_data = args.num_data_inputs
    if num_data is None:
        num_data = _infer_num_data_inputs(inputs_path)
        log.info(f"Auto-detected num_data_inputs={num_data} from inputs.npz")

    # n_steps / n_accum: derive from inputs.npz mini-batch count if not specified
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

    # 8. Build unique_mb_data from npz (only data_size unique samples).
    # The C harness cycles through them via mb % TRAINING_DATA_SIZE.
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

    # Initial weight arrays: npz_base[num_data .. grad_buf_start_idx-1]
    if grad_buf_start_idx > num_data:
        init_weights = list(npz_base[num_data:grad_buf_start_idx])
    else:
        init_weights = []

    # 9. Load reference loss from outputs.npz.
    reference_losses = _load_reference_losses(args.dir)

    # 10. Generate all output files
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

    # 11. Write resolved config for execution.py to pick up after subprocess call.
    meta = {
        "n_train_steps": n_steps,
        "n_accum_steps": n_accum,
        "training_num_data_inputs": num_data,
    }
    meta_path = os.path.join(args.dumpdir, "training_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent = 2)
    log.info(f"Training meta written to {meta_path}: {meta}")


if __name__ == '__main__':
    parser = TestGeneratorArgumentParser(description = "Deeploy Training Code Generation Utility.")
    parser.add_argument("--cores", type = int, default = 1, help = "Number of cluster cores. Default: 1.")
    add_training_inference_args(parser)
    parser.add_argument("--shouldFail", action = "store_true")
    parser.set_defaults(shouldFail = False)
    args = parser.parse_args()

    try:
        generateTrainingNetwork(args)
    except Exception:
        if args.shouldFail:
            print("\033[92mTraining network generation ended, failed as expected!\033[0m")
            sys.exit(0)
        raise
    if args.shouldFail:
        raise RuntimeError("Expected to fail!")
