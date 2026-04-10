# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers used by the training / optimizer code-generation entry points
(generateTrainingNetwork.py, testMVPTraining.py, generateOptimizerNetwork.py,
testMVPOptimizer.py).

Three kinds of helpers live here:

1. inputs.npz / outputs.npz readers (``_load_reference_losses``, ``_infer_*``).
2. The singleton ``_mockScheduler`` the Tiler expects for per-node tiling.
3. argparse builders and the ``--shouldFail`` handshake runner that each
   codegen entry point would otherwise have to duplicate verbatim in its
   ``if __name__ == '__main__':`` block.
"""

import argparse
import os
import sys
from typing import Callable, List, Optional

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.Logging import DEFAULT_LOGGER as log

# Graph input name marker identifying gradient accumulation buffers.
_GRAD_ACC = "_grad.accumulation.buffer"


def _load_reference_losses(train_dir: str) -> Optional[list]:
    """Load reference loss values from outputs.npz.

    Returns the list of per-mini-batch loss values if any key in
    outputs.npz contains 'loss', otherwise None (with a warning).
    """
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
    """Auto-detect number of data inputs from inputs.npz.

    Data inputs are the base arr_* entries that have per-mini-batch
    variants (mb1_arr_*) in the npz — i.e. entries that actually change
    across mini-batches.

    Raises ValueError if no mb1 entries are found (single-mini-batch case)
    where the data/weight boundary cannot be determined automatically.
    """
    inputs = np.load(inputs_path)
    base_keys = sorted(k for k in inputs.files if not k.startswith('mb') and not k.startswith('meta_'))
    count = sum(1 for k in base_keys if f'mb1_{k}' in inputs.files)
    if count == 0:
        raise ValueError("Cannot auto-detect num_data_inputs: inputs.npz has only one mini-batch "
                         "(no mb1_arr_* entries found). Please pass --num-data-inputs explicitly.")
    return count


def _infer_total_mb(inputs_path: str) -> int:
    """Count total mini-batches from inputs.npz.

    New format: inputs.npz contains meta_n_batches (total training mini-batches)
    and meta_data_size (number of unique samples stored; C harness cycles via modulo).

    Legacy format: count 1 + number of unique mb* indices.
    """
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
    """Return the number of unique input samples stored in inputs.npz.

    New format: reads meta_data_size.
    Legacy format: same as _infer_total_mb (all batches were unique).
    """
    inputs = np.load(inputs_path)
    if "meta_data_size" in inputs.files:
        return int(inputs["meta_data_size"].flat[0])
    return _infer_total_mb(inputs_path)


def _infer_n_accum(inputs_path: str) -> int:
    """Return the gradient accumulation step count stored in inputs.npz.

    New format: reads meta_n_accum written by the exporter.
    Legacy format: defaults to 1 (no gradient accumulation).
    """
    inputs = np.load(inputs_path)
    if "meta_n_accum" in inputs.files:
        return int(inputs["meta_n_accum"].flat[0])
    return 1


def _mockScheduler(graph: gs.Graph) -> List[List[gs.Node]]:
    """Wrap every node in a singleton list for the Tiler pattern interface."""
    return [[node] for node in graph.nodes]


# ---------------------------------------------------------------------------
# argparse builders
#
# The four training / optimizer codegen entry points all define the same
# arguments in their __main__ blocks.  These helpers add the shared groups
# to an existing parser so each entry point only has to compose the groups
# it actually needs.
# ---------------------------------------------------------------------------


def add_cores_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cores",
        type = int,
        default = 1,
        help = "Number of cores on which the network is run. Default: 1.",
    )


def add_training_inference_args(parser: argparse.ArgumentParser) -> None:
    """Arguments consumed by both training codegen entry points."""
    parser.add_argument(
        "--num-data-inputs",
        type = int,
        dest = "num_data_inputs",
        default = None,
        help = "Number of DATA inputs that change per mini-batch. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--n-steps",
        type = int,
        dest = "n_steps",
        default = None,
        help = "N_TRAIN_STEPS: number of gradient-accumulation update steps. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--n-accum",
        type = int,
        dest = "n_accum",
        default = None,
        help = "N_ACCUM_STEPS: number of mini-batches per update step. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--learning-rate",
        type = float,
        dest = "learning_rate",
        default = 0.001,
        help = "SGD learning rate emitted as TRAINING_LEARNING_RATE in testinputs.h. Default: 0.001.",
    )
    parser.add_argument(
        "--tolerance",
        type = float,
        dest = "tolerance_abs",
        default = 1e-3,
        help = "Absolute loss tolerance emitted as TRAINING_TOLERANCE_ABS in testoutputs.h. Default: 1e-3.",
    )


def add_memory_level_args(parser: argparse.ArgumentParser) -> None:
    """L1/L2 sizes and the default IO memory level."""
    parser.add_argument(
        "--l1",
        type = int,
        dest = "l1",
        default = 64_000,
        help = "Set L1 size in bytes. Default: 64000.",
    )
    parser.add_argument(
        "--l2",
        type = int,
        dest = "l2",
        default = 1_024_000,
        help = "Set L2 size in bytes. Default: 1024000.",
    )
    parser.add_argument(
        "--defaultMemLevel",
        type = str,
        dest = "defaultMemLevel",
        default = "L2",
        help = "Default memory level for IO buffers. Default: L2.",
    )


def add_tiling_solver_args(parser: argparse.ArgumentParser) -> None:
    """Arguments specific to the tiled codegen path."""
    parser.add_argument(
        "--memAllocStrategy",
        type = str,
        dest = "memAllocStrategy",
        default = "MiniMalloc",
        help = "Memory allocation strategy. Default: MiniMalloc.",
    )
    parser.add_argument(
        "--searchStrategy",
        type = str,
        dest = "searchStrategy",
        default = "random-max",
        help = "CP solver search strategy. Default: random-max.",
    )
    parser.add_argument(
        "--plotMemAlloc",
        action = "store_true",
        help = "Save memory allocation plots in the deeployStates folder.",
    )
    parser.add_argument(
        "--profileTiling",
        action = "store_true",
        help = "Enable tiling profiling (inserts cycle counters around each tiled kernel).",
    )


def add_optimizer_training_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-dir",
        type = str,
        default = None,
        help = "Directory containing the training network.onnx.  When provided, "
        "weight and grad-acc buffers are shared with TrainingNetwork instead "
        "of being allocated independently.",
    )


def add_should_fail_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shouldFail", action = "store_true")
    parser.set_defaults(shouldFail = False)


def run_with_shouldfail(fn: Callable[[argparse.Namespace], None], args: argparse.Namespace,
                        stage_label: str) -> None:
    """Invoke ``fn(args)`` honouring the ``--shouldFail`` handshake.

    On success with ``--shouldFail``: raises ``RuntimeError("Expected to fail!")``.
    On exception with ``--shouldFail``: prints a green success banner and exits 0.
    Otherwise: exception propagates, success returns normally.
    """
    try:
        fn(args)
    except Exception:
        if args.shouldFail:
            print(f"\033[92m{stage_label} ended, failed as expected!\033[0m")
            sys.exit(0)
        raise
    if args.shouldFail:
        raise RuntimeError("Expected to fail!")
