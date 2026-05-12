# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Code-generation helpers for the training / optimizer test harness.

These functions emit the C source, header and data files for training tests
that drive both a TrainingNetwork (forward + backward + gradient accumulation)
and an OptimizerNetwork (SGD weight update) on the target platform.

Kept as a separate module from testUtils.codeGenerate (which handles plain
inference codegen) so this PR's training-side additions touch the inference
helpers only through imports, not by interleaving with inference definitions.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from Deeploy.DeeployTypes import CodeGenVerbosity, NetworkDeployer
from Deeploy.Targets.MemPool.Platform import MemPoolPlatform
from Deeploy.Targets.PULPOpen.Platform import MemoryPULPPlatform, MemoryPULPPlatformWrapper, PULPPlatform

from .codeGenerate import generateL3HexDump


def generateTrainingTestInputsHeader(deployer: NetworkDeployer,
                                     all_mb_data: List[List[np.ndarray]],
                                     n_steps: int,
                                     n_accum: int,
                                     grad_buf_start_idx: int = 0,
                                     num_grad_inputs: int = 0,
                                     learning_rate: float = 0.001,
                                     init_weights: List[np.ndarray] = None,
                                     data_size: int = None) -> str:
    """Generate testinputs.h for training tests.

    Parameters
    ----------
    deployer : NetworkDeployer
        Prepared deployer (used to look up buffer types).
    all_mb_data : list of list of np.ndarray
        Per-mini-batch DATA arrays: ``all_mb_data[mb][buf]`` is the array for
        mini-batch *mb* and DATA buffer *buf*.  All mini-batches must have the
        same number of buffers.
    n_steps : int
        N_TRAIN_STEPS macro value.
    n_accum : int
        N_ACCUM_STEPS macro value.
    grad_buf_start_idx : int
        Index of the first grad accumulation buffer in DeeployNetwork_inputs[].
        Used to emit TRAINING_GRAD_BUF_START_IDX.  Pass 0 (and num_grad_inputs=0)
        to suppress the define (e.g. when no grad bufs exist).
    num_grad_inputs : int
        Number of grad accumulation buffers.  Used to emit TRAINING_NUM_GRAD_INPUTS.

    Returns
    -------
    str
        C header string.
    """
    total_mb = n_steps * n_accum
    num_data = len(all_mb_data[0]) if all_mb_data else 0
    # data_size: number of unique samples stored in C arrays.
    # C harness cycles: testDataVector[mb % TRAINING_DATA_SIZE].
    # Defaults to total_mb (no cycling) for backward compatibility.
    effective_data_size = data_size if (data_size is not None and data_size < total_mb) else total_mb

    retStr = ""
    retStr += f"#define N_TRAIN_STEPS {n_steps}\n"
    retStr += f"#define N_ACCUM_STEPS {n_accum}\n"
    retStr += f"#define TRAINING_DATA_SIZE {effective_data_size}\n"
    retStr += f"#define TRAINING_NUM_DATA_INPUTS {num_data}\n"
    if num_grad_inputs > 0:
        retStr += f"#define TRAINING_GRAD_BUF_START_IDX {grad_buf_start_idx}\n"
        retStr += f"#define TRAINING_NUM_GRAD_INPUTS {num_grad_inputs}\n"
        num_weight_inputs = grad_buf_start_idx - num_data
        retStr += f"#define TRAINING_NUM_WEIGHT_INPUTS {num_weight_inputs}\n"
    retStr += f"#define TRAINING_LEARNING_RATE {learning_rate:.10g}f\n"
    retStr += "\n"

    # Emit per-mini-batch buffer arrays — only effective_data_size unique rows.
    # all_mb_data must contain exactly effective_data_size rows.
    for mb in range(effective_data_size):
        mb_data = all_mb_data[mb] if mb < len(all_mb_data) else all_mb_data[-1]
        row_entries = []
        for buf_idx, arr in enumerate(mb_data):
            values = arr.reshape(-1)

            # Determine C type from deployer context (buffer "input_N").
            input_key = f"input_{buf_idx}"
            if deployer.ctxt.is_buffer(input_key):
                buffer = deployer.ctxt.lookup(input_key)
                typeName = buffer._type.referencedType.typeName
                typeWidth = buffer._type.referencedType.typeWidth
            else:
                # Fallback: infer from numpy dtype
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    typeName = "float32_t"
                    typeWidth = 32
                elif arr.dtype == np.int64:
                    typeName = "int64_t"
                    typeWidth = 64
                elif arr.dtype == np.bool_ or arr.dtype == bool:
                    typeName = "uint8_t"
                    typeWidth = 8
                else:
                    typeName = "int32_t"
                    typeWidth = 32

            buf_name = f"testData_mb{mb}_buf{buf_idx}"
            row_entries.append(buf_name)

            # Format values
            if typeName == 'float32_t':
                list_str = ", ".join(
                    [f'{float(x)}f' if not (np.isinf(x) or np.isnan(x)) else str(x) for x in values.astype(np.float32)])
            else:
                list_str = ", ".join([str(x) for x in values])

            # 4-byte alignment padding
            total_bytes = (values.size * typeWidth) // 8
            pad_bytes = (-total_bytes) % 4
            if pad_bytes:
                paddingElements = (pad_bytes * 8 + typeWidth - 1) // typeWidth
                list_str += ", " + ", ".join("0" for _ in range(paddingElements))

            retStr += f"{typeName} {buf_name}[] = {{{list_str}}};\n"

        # Emit the row pointer array for this mini-batch
        row_name = f"testDataRow{mb}"
        retStr += f"void* {row_name}[] = {{{', '.join(f'(void*){e}' for e in row_entries)}}};\n"
        retStr += "\n"

    # Emit the top-level vector of row pointers (only unique samples; C harness cycles via modulo).
    retStr += f"void** testDataVector[{effective_data_size}] = {{{', '.join(f'testDataRow{mb}' for mb in range(effective_data_size))}}};\n"

    # Emit initial weight arrays (one per weight input, indices num_data..grad_buf_start_idx-1).
    if init_weights:
        retStr += "\n"
        weight_entries = []
        num_data = len(all_mb_data[0]) if all_mb_data else 0
        for wi, arr in enumerate(init_weights):
            buf_global_idx = num_data + wi
            input_key = f"input_{buf_global_idx}"
            if deployer.ctxt.is_buffer(input_key):
                buffer = deployer.ctxt.lookup(input_key)
                typeName = buffer._type.referencedType.typeName
                typeWidth = buffer._type.referencedType.typeWidth
            else:
                typeName = "float32_t"
                typeWidth = 32
            values = arr.reshape(-1).astype(np.float32)
            # Tile values to match Deeploy's internal (possibly sequence-length-tiled) shape.
            if deployer.ctxt.is_buffer(input_key):
                expected_nelems = int(np.prod(deployer.ctxt.lookup(input_key).shape))
                if expected_nelems > len(values) and expected_nelems % len(values) == 0:
                    values = np.tile(values, expected_nelems // len(values))
            list_str = ", ".join([f'{float(x)}f' for x in values])
            buf_name = f"testInitWeight_{wi}"
            weight_entries.append(buf_name)
            # Place initial weight arrays in WEIGHTMEM_SRAM (defined in the
            # Siracusa linker script, ~4 MB on-chip). Two reasons:
            #   1) `.l2_data` is only ~1.94 MB after kernel-reserved space;
            #      MobileNetV1-0.25 testInitWeight_* alone is ~830 KB and
            #      together with the per-mb data (~432 KB) overflows L2 at
            #      link time.
            #   2) Even when the model fits, leaving the weights in `.l2_data`
            #      means the harness must `l3_aware_copy` them into the L3
            #      training buffers at boot — but Deeploy already emits a
            #      `load_file_to_ram(buf, "N.hex")` for every L3-resident
            #      input, and gvsoc's HyperFlash simulation is *very* slow
            #      per-file. With ~80 hex files, the boot stalls 25+ min;
            #      with weights baked into the binary via WEIGHTMEM_SRAM,
            #      gvsoc loads them in one shot with the program image and
            #      ResNet8/MobileNetV1 sim finishes in ~1 min.
            retStr += f'{typeName} {buf_name}[] __attribute__((section(".weightmem_sram"))) = {{{list_str}}};\n'
        retStr += f"void* testInitWeights[{len(weight_entries)}] = {{{', '.join(f'(void*){e}' for e in weight_entries)}}};\n"

    return retStr


def generateTrainingTestOutputsHeader(
    reference_losses: List = None,
    tolerance_abs: float = 1e-3,
) -> str:
    """Generate testoutputs.h for training tests — loss comparison only.

    Parameters
    ----------
    reference_losses : list of float, optional
        Reference loss value for each forward pass (one per mini-batch step).
        If None, loss comparison is skipped.
    tolerance_abs : float
        Absolute comparison tolerance emitted as TRAINING_TOLERANCE_ABS.

    Returns
    -------
    str
        C header string.
    """
    has_loss = reference_losses is not None and len(reference_losses) > 0

    retStr = "// testoutputs.h — Phase 2: loss verification\n"
    retStr += f"#define TRAINING_TOLERANCE_ABS {tolerance_abs:.10g}f\n\n"

    if has_loss:
        n = len(reference_losses)
        retStr += "// Expected loss for each forward pass (one per mini-batch)\n"
        retStr += f"#define N_LOSS_REFS {n}\n"
        vals = ", ".join(f"{float(v):.10g}f" for v in reference_losses)
        retStr += f"float32_t testLossRef[{n}] = {{{vals}}};\n\n"
    else:
        retStr += "// No loss reference available — loss comparison skipped.\n"
        retStr += "#define N_LOSS_REFS 0\n\n"

    return retStr


def generateTrainingNetworkHeader(deployer: NetworkDeployer) -> str:
    """Generate TrainingNetwork.h — same as generateTestNetworkHeader but with
    RunTrainingNetwork / InitTrainingNetwork function names and a distinct header guard.

    Parameters
    ----------
    deployer : NetworkDeployer
        Prepared deployer.

    Returns
    -------
    str
        C header string.
    """
    retStr = ""

    retStr += """
#ifndef __DEEPLOY_TRAINING_HEADER__
#define __DEEPLOY_TRAINING_HEADER__
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
"""
    retStr += deployer.generateIncludeString()
    if isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
void RunTrainingNetwork();
void InitTrainingNetwork();

"""
    else:
        retStr += """
void RunTrainingNetwork(uint32_t core_id, uint32_t numThreads);
void InitTrainingNetwork(uint32_t core_id, uint32_t numThread);

"""

    retStr += deployer.generateIOBufferInitializationCode()
    retStr += """
#endif
"""

    return retStr


def generateTrainingNetworkImplementation(deployer: NetworkDeployer, verbosityCfg: CodeGenVerbosity) -> str:
    """Generate TrainingNetwork.c — same as generateTestNetworkImplementation but with
    RunTrainingNetwork / InitTrainingNetwork function names and including TrainingNetwork.h.

    Parameters
    ----------
    deployer : NetworkDeployer
        Prepared deployer.
    verbosityCfg : CodeGenVerbosity
        Verbosity configuration.

    Returns
    -------
    str
        C implementation string.
    """
    retStr = ""

    retStr += """#include <stdio.h>
#include <stdlib.h>
#include <math.h>
"""
    retStr += deployer.generateIncludeString()
    retStr += """

#include "TrainingNetwork.h"

"""

    retStr += deployer.generateBufferInitializationCode()
    retStr += deployer.generateGlobalDefinitionCode()

    if isinstance(deployer.Platform, MemPoolPlatform):
        retStr += deployer.generateInferenceInitializationCode()
        retStr += """
void RunTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
"""
    elif isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
void RunTrainingNetwork(){
"""
        retStr += deployer.generateInferenceInitializationCode()
    else:
        retStr += """
void RunTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
"""
        retStr += deployer.generateInferenceInitializationCode()

    retStr += deployer.generateFunction(verbosityCfg)
    if isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
}

void InitTrainingNetwork(){
"""
    else:
        retStr += """
}

void InitTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
"""
    retStr += deployer.generateEngineInitializationCode()
    retStr += deployer.generateBufferAllocationCode()
    retStr += """
}
"""

    return retStr


def generateTrainingTestNetwork(deployer: NetworkDeployer,
                                all_mb_data: List[List[np.ndarray]],
                                dumpdir: str,
                                verbosityCfg: CodeGenVerbosity,
                                n_steps: int = 1,
                                n_accum: int = 1,
                                num_data_inputs: int = 2,
                                grad_buf_start_idx: int = 0,
                                num_grad_inputs: int = 0,
                                learning_rate: float = 0.001,
                                reference_losses: List = None,
                                init_weights: List = None,
                                data_size: int = None,
                                tolerance_abs: float = 1e-3) -> None:
    """Generate all training test files: testinputs.h, testoutputs.h, TrainingNetwork.h, TrainingNetwork.c.

    Parameters
    ----------
    deployer : NetworkDeployer
        Prepared deployer (ctxt.name must already be set to "DeeployTrainingNetwork").
    all_mb_data : list of list of np.ndarray
        Per-mini-batch DATA arrays: ``all_mb_data[mb][buf]`` is the array for
        mini-batch *mb* and DATA buffer *buf*.
    dumpdir : str
        Output directory for generated files.
    verbosityCfg : CodeGenVerbosity
        Verbosity configuration.
    n_steps : int
        N_TRAIN_STEPS value.
    n_accum : int
        N_ACCUM_STEPS value.
    num_data_inputs : int
        Number of data inputs (TRAINING_NUM_DATA_INPUTS).
    grad_buf_start_idx : int
        Index of the first grad accumulation buffer in DeeployNetwork_inputs[].
    num_grad_inputs : int
        Number of grad accumulation buffers (TRAINING_NUM_GRAD_INPUTS).
    """
    assert deployer.prepared, "An unprepared deployer was given"

    os.makedirs(dumpdir, exist_ok = True)

    # testinputs.h
    testInputStr = generateTrainingTestInputsHeader(deployer,
                                                    all_mb_data,
                                                    n_steps,
                                                    n_accum,
                                                    grad_buf_start_idx,
                                                    num_grad_inputs,
                                                    learning_rate,
                                                    init_weights = init_weights,
                                                    data_size = data_size)
    with open(f'{dumpdir}/testinputs.h', 'w') as f:
        f.write(testInputStr)

    # testoutputs.h
    testOutputStr = generateTrainingTestOutputsHeader(
        reference_losses = reference_losses,
        tolerance_abs = tolerance_abs,
    )
    with open(f'{dumpdir}/testoutputs.h', 'w') as f:
        f.write(testOutputStr)

    # TrainingNetwork.h
    headerStr = generateTrainingNetworkHeader(deployer)
    with open(f'{dumpdir}/TrainingNetwork.h', 'w') as f:
        f.write(headerStr)

    # TrainingNetwork.c
    implStr = generateTrainingNetworkImplementation(deployer, verbosityCfg)
    with open(f'{dumpdir}/TrainingNetwork.c', 'w') as f:
        f.write(implStr)

    clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
    for fname in ['TrainingNetwork.c', 'TrainingNetwork.h', 'testinputs.h', 'testoutputs.h']:
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/{fname}')

    # Build initial-value list for every input_N buffer so that L3 hex files
    # can be written.  The list must cover all N where "input_N" exists in the
    # deployer context.  Layout (must match DeeployNetwork_inputs[] order):
    #   [0 .. num_data_inputs-1]              → first mini-batch data
    #   [num_data_inputs .. grad_start-1]     → initial weights
    #   [grad_start .. grad_start+num_grad-1] → zeros  (grad acc bufs)
    #   [last]                                → lazy_reset_grad = 1 (uint8)
    l3_initial_inputs: List[np.ndarray] = []
    # Count how many input_N buffers exist in the deployer context
    n_total_inputs = sum(
        1 for name in deployer.ctxt.globalObjects if name.startswith("input_") and name[len("input_"):].isdigit())
    for i in range(n_total_inputs):
        if all_mb_data and i < len(all_mb_data[0]):
            # Data / label input
            l3_initial_inputs.append(all_mb_data[0][i])
        elif (init_weights is not None and grad_buf_start_idx > 0 and num_data_inputs <= i < grad_buf_start_idx):
            # Weight input
            wi = i - num_data_inputs
            l3_initial_inputs.append(init_weights[wi] if wi <
                                     len(init_weights) else np.array([0.0], dtype = np.float32))
        elif (grad_buf_start_idx > 0 and num_grad_inputs > 0
              and grad_buf_start_idx <= i < grad_buf_start_idx + num_grad_inputs):
            # Gradient accumulation buffer — zero-initialised
            buf = deployer.ctxt.globalObjects.get(f"input_{i}")
            shape = buf.shape if (buf is not None and hasattr(buf, 'shape')) else (1,)
            l3_initial_inputs.append(np.zeros(shape, dtype = np.float32))
        else:
            # lazy_reset_grad (last input) or any unknown slot — default 1 / uint8
            buf = deployer.ctxt.globalObjects.get(f"input_{i}")
            shape = buf.shape if (buf is not None and hasattr(buf, 'shape')) else (1,)
            l3_initial_inputs.append(np.ones(shape, dtype = np.uint8))

    generateL3HexDump(deployer, os.path.join(dumpdir, 'hex'), l3_initial_inputs, [])


# ---------------------------------------------------------------------------
# Optimizer network code-generation helpers
# ---------------------------------------------------------------------------

_OPT_PREFIX = "DeeployOptNetwork_"
_TRAIN_PREFIX = "DeeployNetwork_"


def build_shared_buffer_maps(train_onnx_path: str, opt_onnx_model) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build optimizer→training index maps for tensors shared between the two graphs.

    The optimizer ONNX inputs are interleaved weight/grad pairs that have the
    same tensor names as inputs in the training ONNX graph.  We match by name
    so that ``InitOptimizerNetwork`` can reference the already-allocated
    ``DeeployNetwork_input_N`` pointers instead of allocating fresh buffers.

    Parameters
    ----------
    train_onnx_path : str
        Path to the training ``network.onnx``.
    opt_onnx_model :
        Already-loaded optimizer ONNX model (``onnx.ModelProto``).

    Returns
    -------
    shared_input_map : Dict[int, int]
        opt_input_idx → train_input_idx
    shared_output_map : Dict[int, int]
        opt_output_idx → train_input_idx  (SGD outputs == updated weights,
        same physical buffer as the weight input)
    """
    import onnx as _onnx
    train_model = _onnx.load_model(train_onnx_path)
    train_names = [inp.name for inp in train_model.graph.input]
    train_name_to_idx = {name: i for i, name in enumerate(train_names)}

    opt_input_names = [inp.name for inp in opt_onnx_model.graph.input]
    opt_output_names = [out.name for out in opt_onnx_model.graph.output]

    shared_input_map: Dict[int, int] = {}
    for opt_idx, name in enumerate(opt_input_names):
        if name in train_name_to_idx:
            shared_input_map[opt_idx] = train_name_to_idx[name]

    shared_output_map: Dict[int, int] = {}
    for opt_idx, name in enumerate(opt_output_names):
        # Try exact match first; then strip the '_updated' suffix that the SGD
        # node appends to output tensor names (e.g. 'conv1_weight_updated' → 'conv1_weight').
        lookup_name = name
        if lookup_name not in train_name_to_idx and lookup_name.endswith('_updated'):
            lookup_name = lookup_name[:-len('_updated')]
        if lookup_name in train_name_to_idx:
            shared_output_map[opt_idx] = train_name_to_idx[lookup_name]

    return shared_input_map, shared_output_map


def _patch_shared_buffers(retStr: str, shared_input_map: Dict[int, int], shared_output_map: Dict[int, int]) -> str:
    """Redirect optimizer I/O buffers to Training's already-allocated buffers.

    Must be called AFTER the _TRAIN_PREFIX → _OPT_PREFIX substitution so that
    the generated symbols already carry the ``DeeployOptNetwork_`` prefix.

    Handles two allocation styles produced by Deeploy:

    *Non-tiled* (per-buffer malloc)::

        DeeployOptNetwork_input_N = (SomeType *)pi_l2_malloc(sizeof(...));

    *Tiled* (single arena with offsets)::

        DeeployOptNetwork_input_N = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + OFFSET);

    Both are replaced with direct pointers into the TrainingNetwork arenas::

        DeeployOptNetwork_input_N = (float32_t *)DeeployNetwork_input_M;

    After all I/O pointers are redirected, if a ``MEMORYARENA_L2`` or
    ``MEMORYARENA_L3`` allocation is no longer referenced anywhere in the Init
    body (i.e., the shared buffers consumed the entire arena), the now-unused
    malloc is also removed to reclaim the L2/L3 memory.

    Parameters
    ----------
    retStr : str
        The already-prefix-substituted C source string.
    shared_input_map : Dict[int, int]
        Optimizer input index → training input index.
    shared_output_map : Dict[int, int]
        Optimizer output index → training input index (in-place update).

    Returns
    -------
    str
        Patched C source string.
    """
    if not shared_input_map and not shared_output_map:
        return retStr

    # ------------------------------------------------------------------
    # Pattern 1 (non-tiled): individual pi_*_malloc per buffer
    # ------------------------------------------------------------------
    _malloc_pat = re.compile(
        r'(DeeployOptNetwork_(input|output)_(\d+))\s*=\s*\([^)]+\s*\*\s*\)\s*pi_\w+_malloc\([^;]+\);')

    # ------------------------------------------------------------------
    # Pattern 2 (tiled): arena-offset assignment
    #   DeeployOptNetwork_input_N = (Type *)((char *)DeeployOptNetwork_MEMORYARENA_Lx + OFFSET);
    # ------------------------------------------------------------------
    _arena_pat = re.compile(r'(DeeployOptNetwork_(input|output)_(\d+))\s*=\s*\([^)]+\s*\*\s*\)'
                            r'\s*\(\s*\(char\s*\*\)\s*DeeployOptNetwork_MEMORYARENA_L\w+\s*\+\s*\d+\s*\)\s*;')

    def _make_replacement(symbol: str, kind: str, idx: int) -> Optional[str]:
        if kind == "input" and idx in shared_input_map:
            train_idx = shared_input_map[idx]
            return f'{symbol} = (float32_t *){_TRAIN_PREFIX}input_{train_idx};  /* shared with TrainingNetwork */'
        if kind == "output" and idx in shared_output_map:
            train_idx = shared_output_map[idx]
            return f'{symbol} = (float32_t *){_TRAIN_PREFIX}input_{train_idx};  /* in-place, shared with TrainingNetwork */'
        return None

    def _replace(m: re.Match) -> str:
        replacement = _make_replacement(m.group(1), m.group(2), int(m.group(3)))
        return replacement if replacement is not None else m.group(0)

    retStr = _malloc_pat.sub(_replace, retStr)
    retStr = _arena_pat.sub(_replace, retStr)

    # ------------------------------------------------------------------
    # Arena elimination: if a MEMORYARENA_Lx is no longer used for any
    # pointer arithmetic after the redirects, its malloc is dead and can
    # be removed to reclaim L2/L3.  The global declaration is left in
    # place (harmless; the variable will be NULL at runtime).
    # ------------------------------------------------------------------
    for level in ('L2', 'L3'):
        arena_sym = f'DeeployOptNetwork_MEMORYARENA_{level}'
        # Pattern for the malloc assignment line itself
        malloc_line_pat = re.compile(
            rf'[^\n]*{re.escape(arena_sym)}\s*=\s*\([^)]+\)\s*(?:pi_\w+_malloc|cl_ram_malloc)\([^;]+\);\s*\n')
        # Pattern for any use of the arena in pointer arithmetic:
        #   (char *)ARENA + OFFSET  or  (void *)ARENA  etc.
        arena_use_pat = re.compile(rf'\(\s*(?:char|void|int8_t)\s*\*\s*\)\s*{re.escape(arena_sym)}')
        if not arena_use_pat.search(retStr):
            # No remaining pointer arithmetic — the malloc is dead
            retStr = malloc_line_pat.sub('', retStr)

    # ------------------------------------------------------------------
    # Inject TrainingNetwork header so DeeployNetwork_input_N symbols resolve
    # ------------------------------------------------------------------
    retStr = retStr.replace(
        '#include "OptimizerNetwork.h"',
        '#include "OptimizerNetwork.h"\n#include "TrainingNetwork.h"',
    )
    return retStr


def _patch_shared_arenas(retStr: str, train_c_source: str) -> str:
    """Redirect optimizer L1/L2 arena allocations to reuse training network's arenas.

    TrainingNetwork and OptimizerNetwork run strictly sequentially: RunTrainingNetwork()
    completes before RunOptimizerNetwork() starts.  Their L1/L2 tile-working arenas
    therefore never overlap in time and can share the same physical memory.

    Only the L1 arena is shared: it is pure tile-compute scratch whose content is
    dead after each kernel returns.  The L2 arena is NOT shared because it may hold
    persistent tensor data (weights, activations) at fixed offsets in non-tiled mode;
    sharing it would let the optimizer's L2 staging buffers overwrite that data.

    Must be called AFTER the _TRAIN_PREFIX → _OPT_PREFIX substitution.

    Parameters
    ----------
    retStr : str
        The already-prefix-substituted C source string for the optimizer.
    train_c_source : str
        The full text of TrainingNetwork.c (used to confirm the arena symbols exist).

    Returns
    -------
    str
        Patched C source string.
    """
    for level in ('L1',):
        train_sym = f'DeeployNetwork_MEMORYARENA_{level}'
        # Only alias if the training network actually has this arena
        if train_sym not in train_c_source:
            continue

        opt_sym = f'DeeployOptNetwork_MEMORYARENA_{level}'
        opt_malloc_pat = re.compile(rf'({re.escape(opt_sym)})\s*=\s*\([^)]+\)\s*\w+\(sizeof\([^)]+\)\s*\*\s*\d+\)\s*;')
        if not opt_malloc_pat.search(retStr):
            continue

        replacement = f'{opt_sym} = (int8_t *){train_sym};  /* shared with TrainingNetwork */'
        retStr = opt_malloc_pat.sub(replacement, retStr)

    # Inject TrainingNetwork header if not already present
    # (_patch_shared_buffers may have already added it; guard against duplicates)
    if '#include "TrainingNetwork.h"' not in retStr:
        retStr = retStr.replace(
            '#include "OptimizerNetwork.h"',
            '#include "OptimizerNetwork.h"\n#include "TrainingNetwork.h"',
        )

    return retStr


def _ensure_training_l1_capacity(dumpdir: str, train_c_source: str, opt_alloc_code: str) -> str:
    """Enlarge TrainingNetwork's L1 arena to cover the optimizer's L1 needs.

    Since the two networks share the same L1 arena, TrainingNetwork must allocate
    at least max(train_L1, opt_L1) bytes.  When the optimizer needs more L1 than
    training (rare but possible, e.g. autoencoder), this function patches
    TrainingNetwork.c and TrainingNetwork.h in-place and returns the updated
    TrainingNetwork.c source string.

    Parameters
    ----------
    dumpdir : str
        Directory containing TrainingNetwork.c and TrainingNetwork.h.
    train_c_source : str
        Current content of TrainingNetwork.c.
    opt_alloc_code : str
        Optimizer buffer-allocation code after _TRAIN_PREFIX → _OPT_PREFIX
        substitution (used to extract the optimizer's L1 size).

    Returns
    -------
    str
        (Possibly updated) TrainingNetwork.c source string.
    """
    m_opt = re.search(
        r'DeeployOptNetwork_MEMORYARENA_L1\s*=\s*\([^)]+\)\s*pmsis_l1_malloc\(sizeof\([^)]+\)\s*\*\s*(\d+)\)',
        opt_alloc_code,
    )
    if not m_opt:
        return train_c_source

    opt_l1 = int(m_opt.group(1))

    m_train = re.search(
        r'(DeeployNetwork_MEMORYARENA_L1\s*=\s*\([^)]+\)\s*pmsis_l1_malloc\(sizeof\([^)]+\)\s*\*\s*)(\d+)(\))',
        train_c_source,
    )
    if not m_train:
        return train_c_source

    train_l1 = int(m_train.group(2))
    if opt_l1 <= train_l1:
        return train_c_source  # Already large enough

    new_l1 = opt_l1

    # Patch TrainingNetwork.c malloc size
    train_c_new = train_c_source.replace(
        m_train.group(0),
        f'{m_train.group(1)}{new_l1}{m_train.group(3)}',
        1,
    )
    train_c_path = os.path.join(dumpdir, 'TrainingNetwork.c')
    with open(train_c_path, 'w') as f:
        f.write(train_c_new)

    # Patch TrainingNetwork.h _len constant
    train_h_path = os.path.join(dumpdir, 'TrainingNetwork.h')
    if os.path.exists(train_h_path):
        train_h = open(train_h_path).read()
        train_h_new = re.sub(
            r'(DeeployNetwork_MEMORYARENA_L1_len\s*=\s*)\d+',
            rf'\g<1>{new_l1}',
            train_h,
        )
        with open(train_h_path, 'w') as f:
            f.write(train_h_new)

    return train_c_new


def generateOptimizerNetworkHeader(deployer: NetworkDeployer) -> str:
    """Generate OptimizerNetwork.h.

    Reuses the Deeploy deployer's output and applies two transformations:
      1. Replace the buffer prefix ``DeeployNetwork_`` → ``DeeployOptNetwork_``
      2. Inject ``RunOptimizerNetwork`` / ``InitOptimizerNetwork`` function declarations.

    Parameters
    ----------
    deployer : NetworkDeployer
        Prepared deployer for the optimizer ONNX graph.

    Returns
    -------
    str
        C header string.
    """
    retStr = ""
    retStr += """
#ifndef __DEEPLOY_OPTIMIZER_HEADER__
#define __DEEPLOY_OPTIMIZER_HEADER__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
"""
    retStr += deployer.generateIncludeString()
    if isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
void RunOptimizerNetwork();
void InitOptimizerNetwork();

"""
    else:
        retStr += """
void RunOptimizerNetwork(uint32_t core_id, uint32_t numThreads);
void InitOptimizerNetwork(uint32_t core_id, uint32_t numThreads);

"""
    retStr += deployer.generateIOBufferInitializationCode()
    retStr += """
#endif
"""
    # Prefix substitution: all Deeploy-generated DeeployNetwork_ → DeeployOptNetwork_
    retStr = retStr.replace(_TRAIN_PREFIX, _OPT_PREFIX)
    return retStr


def generateOptimizerNetworkImplementation(deployer: NetworkDeployer,
                                           verbosityCfg: CodeGenVerbosity,
                                           shared_input_map: Optional[Dict[int, int]] = None,
                                           shared_output_map: Optional[Dict[int, int]] = None,
                                           train_c_source: Optional[str] = None) -> str:
    """Generate OptimizerNetwork.c.

    Parameters
    ----------
    deployer : NetworkDeployer
        Prepared deployer for the optimizer ONNX graph.
    verbosityCfg : CodeGenVerbosity
        Verbosity configuration.
    shared_input_map : Dict[int, int], optional
        Optimizer input index → training input index for shared weight/grad buffers.
        When provided, those malloc calls are replaced with references to the
        already-allocated TrainingNetwork buffers.
    shared_output_map : Dict[int, int], optional
        Optimizer output index → training input index for in-place shared outputs.
    train_c_source : str, optional
        Full text of TrainingNetwork.c.  When provided, the optimizer's L1/L2 arena
        malloc calls are replaced with direct pointers to the training arenas,
        saving one L1 and one L2 allocation (safe because the two networks run
        strictly sequentially).

    Returns
    -------
    str
        C implementation string.
    """
    retStr = ""
    retStr += """#include <math.h>
#include <stdio.h>
#include <stdlib.h>
"""
    retStr += deployer.generateIncludeString()
    retStr += """
#include "OptimizerNetwork.h"

"""
    retStr += deployer.generateBufferInitializationCode()
    retStr += deployer.generateGlobalDefinitionCode()

    if isinstance(deployer.Platform, MemPoolPlatform):
        retStr += deployer.generateInferenceInitializationCode()
        retStr += """
void RunOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
"""
    elif isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
void RunOptimizerNetwork(){
"""
        retStr += deployer.generateInferenceInitializationCode()
    else:
        retStr += """
void RunOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
"""
        retStr += deployer.generateInferenceInitializationCode()

    retStr += deployer.generateFunction(verbosityCfg)

    if isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
}

void InitOptimizerNetwork(){
"""
    else:
        retStr += """
}

void InitOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
"""
    retStr += deployer.generateEngineInitializationCode()
    retStr += deployer.generateBufferAllocationCode()
    retStr += """
}
"""
    # Prefix substitution
    retStr = retStr.replace(_TRAIN_PREFIX, _OPT_PREFIX)
    # Replace malloc calls for shared weight/grad buffers with Training pointers
    retStr = _patch_shared_buffers(retStr, shared_input_map or {}, shared_output_map or {})
    # Redirect optimizer L1/L2 arena mallocs to reuse training arenas
    if train_c_source:
        retStr = _patch_shared_arenas(retStr, train_c_source)
    return retStr


def generateOptimizerTestNetwork(deployer: NetworkDeployer,
                                 dumpdir: str,
                                 verbosityCfg: CodeGenVerbosity,
                                 shared_input_map: Optional[Dict[int, int]] = None,
                                 shared_output_map: Optional[Dict[int, int]] = None) -> None:
    """Generate OptimizerNetwork.h and OptimizerNetwork.c.

    Parameters
    ----------
    deployer : NetworkDeployer
        Prepared deployer for the optimizer ONNX graph.
    dumpdir : str
        Output directory for generated files.
    verbosityCfg : CodeGenVerbosity
        Verbosity configuration.
    shared_input_map : Dict[int, int], optional
        Optimizer input index → training input index for shared weight/grad buffers.
    shared_output_map : Dict[int, int], optional
        Optimizer output index → training input index for in-place shared outputs.
    """
    assert deployer.prepared, "An unprepared deployer was given"

    os.makedirs(dumpdir, exist_ok = True)

    train_c_path = os.path.join(dumpdir, 'TrainingNetwork.c')
    train_c_source: Optional[str] = None
    if os.path.exists(train_c_path):
        with open(train_c_path, 'r') as f:
            train_c_source = f.read()

    # Enlarge training L1 arena if optimizer needs more (so unconditional L1 sharing is safe)
    if train_c_source:
        opt_alloc_preview = deployer.generateBufferAllocationCode().replace(_TRAIN_PREFIX, _OPT_PREFIX)
        train_c_source = _ensure_training_l1_capacity(dumpdir, train_c_source, opt_alloc_preview)

    headerStr = generateOptimizerNetworkHeader(deployer)
    with open(f'{dumpdir}/OptimizerNetwork.h', 'w') as f:
        f.write(headerStr)

    implStr = generateOptimizerNetworkImplementation(deployer, verbosityCfg, shared_input_map, shared_output_map,
                                                     train_c_source)
    with open(f'{dumpdir}/OptimizerNetwork.c', 'w') as f:
        f.write(implStr)

    clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
    for fname in ['OptimizerNetwork.c', 'OptimizerNetwork.h']:
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/{fname}')
