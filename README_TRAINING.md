<div align="center">

**[Inference](README.md) · [On-Device Training (this page)](README_TRAINING.md)**

</div>

# On-Device Training with Deeploy

This document describes the **end-to-end training extension** built on top of Deeploy, targeting the [Siracusa](https://arxiv.org/abs/2312.14750) SoC simulated via GVSoC. It covers the architecture, the supported model zoo, how to run a training test, and how to add a new model.

---

## Overview

Deeploy's inference pipeline compiles an ONNX graph into tiled C code. The training extension adds a second compilation stage: given an ONNX *training graph* (forward + backward ops, gradient accumulation, and a separate optimizer subgraph), Deeploy generates:

| Network | Entry points | What it does |
|---|---|---|
| **TrainingNetwork** | `InitTrainingNetwork` / `RunTrainingNetwork` | Forward pass + backward pass + `InPlaceAccumulatorV2` gradient accumulation |
| **OptimizerNetwork** | `InitOptimizerNetwork` / `RunOptimizerNetwork` | SGD weight update (compiled as a second Deeploy graph) |

The Siracusa C harness (`DeeployTest/Platforms/Siracusa/src/deeploytraintest.c`) orchestrates the two networks in the standard accumulate-then-update loop:

```
InitTrainingNetwork()
InitOptimizerNetwork()

for update_step in [0, N_TRAIN_STEPS):
    for accum_step in [0, N_ACCUM_STEPS):
        lazy_reset_grad = (accum_step == 0)   # reset on first, accumulate on rest
        load mini-batch data
        RunTrainingNetwork()                   # fwd + bwd + grad acc
        store loss
    RunOptimizerNetwork()                      # SGD weight update
```

All memory transfers between L2 and L3 (HyperRAM) are handled transparently by the `l3_aware_copy` helper using `ram_read` / `ram_write`.

---

## Memory Layout

### TrainingNetwork inputs

```
[0 .. TRAINING_NUM_DATA_INPUTS-1]            data + labels  (change every mini-batch)
[TRAINING_NUM_DATA_INPUTS ..
 .. TRAINING_GRAD_BUF_START_IDX-1]           trainable weights  (persistent, updated after each optimizer step)
[TRAINING_GRAD_BUF_START_IDX ..
 .. +TRAINING_NUM_GRAD_INPUTS-1]             gradient accumulation buffers  (persistent, zeroed at start)
[DeeployNetwork_num_inputs-1]                lazy_reset_grad  (uint8, set by harness)
```

### OptimizerNetwork inputs and outputs

Inputs (interleaved weight + grad pairs, one pair per trainable parameter):

```
DeeployOptNetwork_inputs[2*i]     weight_i      ← TrainingNetwork weight buffer i
DeeployOptNetwork_inputs[2*i+1]   grad_acc_i    ← TrainingNetwork grad buffer i
```

Outputs (one updated weight per parameter):

```
DeeployOptNetwork_outputs[i]      weight_i_updated  → copied back to TrainingNetwork weight buffer i
```

If the codegen detects that the optimizer's output buffer already aliases the training network's weight buffer (same pointer), the copy is skipped and the update is in-place.

---

## Supported Models

All models run on **Siracusa / GVSoC** and are validated by loss comparison against ORT reference values.

### L2-resident (weights + activations fit in L2)

| Model | L1 size | Steps | Tolerance | Notes |
|---|---|---|---|---|
| **SimpleMLP** | 64 KB | 4 | 1e-3 (default) | Baseline sanity check |
| **Autoencoder-tiny** | 128 KB | 4 | 1e-3 (default) | Bit-exact |
| **DSCNN-XS** | 128 KB | 4 | 1e-3 (default) | MLPerf keyword-spotting backbone |

### L3-spill (weights or activations overflow L2; tiled through HyperRAM)

| Model | L1 size | Steps | Tolerance | Notes |
|---|---|---|---|---|
| **ResNet8** | 128 KB | 4 | 1e-3 (default) | Bit-exact (0/4 errors) |
| **MobileNetV1-0.25** | 128 KB | 4 | 5e-3 | FP32 parallel-reduction noise; per-gradient verification confirms no kernel bug |
| **CCT** | 128 KB | 4 | 5e-3 | Step-0 forward drift ~1.5e-3 (FP reduction order on attention) |
| **CCT-LoRA** | 128 KB | 4 | 1e-3 (default) | 4-step test (2 optimizer steps × 2 accum steps); max diff 2.2e-5 |

---

## Running a Training Test

From the `DeeployTest/` directory:

```bash
# Code-generate, build, and simulate a single training model
python deeployTrainingRunner_tiled_siracusa.py \
    -t Tests/Models/Training/ResNet8/resnet8_train \
    --cores 8 --l1 128000 --l2 2000000 \
    --defaultMemLevel L3

# Run the full CI training suite via pytest
pytest test_platforms.py -m 'siracusa_tiled and training' -v
```

Key CLI flags:

| Flag | Default | Meaning |
|---|---|---|
| `-t` | — | Path to the training test directory (required) |
| `--cores` | `8` | Number of PULP cluster cores |
| `--l1` | `64000` | L1 scratchpad size in bytes |
| `--l2` | `1024000` | L2 budget in bytes |
| `--defaultMemLevel` | `L2` | `L2` keeps all buffers in L2; `L3` spills weights/activations to HyperRAM |
| `--n-steps` | auto | `N_TRAIN_STEPS`: optimizer steps (auto-detected from `inputs.npz`) |
| `--n-accum` | auto | `N_ACCUM_STEPS`: mini-batches per update step (auto-detected) |
| `--num-data-inputs` | auto | Number of data inputs that change per mini-batch; required when there is only one mini-batch in `inputs.npz` |
| `--optimizer-dir` | auto | Directory containing the optimizer `network.onnx`; default derived by replacing `_train` with `_optimizer` |
| `--tolerance` | `1e-3` | Absolute loss tolerance for pass/fail; overrides the value in `TRAINING_MODEL_OVERRIDES` |
| `--memAllocStrategy` | `MiniMalloc` | Memory allocation strategy (`MiniMalloc`, `TetrisRandom`, `TetrisCo-Opt`) |
| `--searchStrategy` | `random-max` | CP solver search strategy (`random-max`, `max`, `min`) |
| `--doublebuffer` | off | Enable double-buffering for DMA transfers |
| `--skipgen` | off | Skip code generation and reuse existing `TEST_SIRACUSA/` build |
| `--skipsim` | off | Skip GVSoC simulation (code-gen and build only) |

The runner generates code in `DeeployTest/TEST_SIRACUSA/Tests/Models/Training/<model>/` and calls CMake + GVSoC automatically.

---

## Test Artifacts

Each training test directory must contain:

```
Tests/Models/Training/<model_name>/<test_name>/
    network.onnx      # training graph: forward + backward + InPlaceAccumulatorV2
    inputs.npz        # arr_NNNN keys: data, labels, weights, grad-acc bufs,
                      #   lazy_reset_grad; mb1_arr_* for multi-mini-batch tests
    outputs.npz       # 'loss' key: reference loss values per mini-batch

<model_name>/<optimizer_name>/  (or symlink to shared optimizer)
    network.onnx      # optimizer graph (SGD step as Deeploy inference graph)
    inputs.npz        # weight + grad-acc init values
    outputs.npz       # updated weight reference (optional)
```

Artifacts are generated by [Onnx4Deeploy](https://github.com/runwangdl/Onnx4Deeploy). To regenerate CCT-LoRA artifacts for 4 steps with gradient accumulation over 2 mini-batches:

```bash
cd /path/to/Onnx4Deeploy
python3 Onnx4Deeploy.py -model CCT -mode train --use-lora --n-batches 4 --n-accum 2
```

---

## Adding a New Model

1. **Export training artifacts** via Onnx4Deeploy (or by hand):
   - `network.onnx` — training graph with `lazy_reset_grad` as a graph input
   - `inputs.npz` — `arr_NNNN` keys (use `repack_singlestep.py` if you have named keys)
   - `outputs.npz` — `loss` key with ORT reference losses

2. **Create the test directory** under `DeeployTest/Tests/Models/Training/<Model>/`.

3. **Register in `test_siracusa_tiled_config.py`**:
   ```python
   # weights + activations fit in L2:
   L2_SINGLEBUFFER_TRAINING_MODELS["Models/Training/MyModel/mymodel_train"] = [64000]

   # weights or activations spill to L3:
   L3_SINGLEBUFFER_TRAINING_MODELS["Models/Training/MyModel/mymodel_train"] = [128000]
   ```
   Add a `TRAINING_MODEL_OVERRIDES` entry only if you need a non-default tolerance or `num_data_inputs`.

4. **Run locally** before opening a PR:
   ```bash
   pre-commit run --all-files   # lint check
   pytest test_platforms.py -k 'mymodel' -v -s
   ```

---

## Architecture Notes

### Two-network compilation

The training graph and optimizer graph are compiled as two independent Deeploy deployments. The harness connects them at runtime by copying pointers into and out of `DeeployOptNetwork_inputs[]`.

### Tiling

Both L2 and L3 training tests use the standard Deeploy tiling pipeline (`TilerDeployerWrapper`). The tiler must detect whether a node's L1 closure is the outermost loop (L2-only path) or will be wrapped by an outer L3 loop. This is determined via `nodeMemoryConstraint.tensorMemoryConstraints`: any `"L3"` entry means an outer `PULPL3Tiling` loop will be emitted.

### Gradient accumulation

`InPlaceAccumulatorV2` accumulates gradients in-place across mini-batches. The harness passes `lazy_reset_grad = 1` on the first mini-batch of each optimizer step (zeroes the accumulator before writing) and `0` on subsequent mini-batches (adds to existing values).

### FP32 precision on the PULP cluster

The 8-core PULP cluster sums partial results in non-deterministic order, introducing ~5 × 10⁻⁴ absolute rounding noise per step. This is normal and not a kernel bug. Over multiple SGD steps the rounding accumulates; the `tolerance` field in `TRAINING_MODEL_OVERRIDES` gates the acceptable per-loss-value drift.

---

## Key Files

| Path | Role |
|---|---|
| `DeeployTest/deeployTrainingRunner_tiled_siracusa.py` | Top-level CLI entry point for tiled Siracusa training |
| `DeeployTest/testMVPTraining.py` | Core code-generation logic for training graphs |
| `DeeployTest/testMVPOptimizer.py` | Code-generation logic for the optimizer graph |
| `DeeployTest/testUtils/trainingUtils.py` | Shared helpers: inputs.npz readers, argparse extensions, CMake flag injection |
| `DeeployTest/testUtils/codeGenerateTraining.py` | Low-level codegen: `testInitWeights`, `testLossRef`, `testoutputs.h` |
| `DeeployTest/Platforms/Siracusa/src/deeploytraintest.c` | C harness: training loop, optimizer step, loss verification |
| `DeeployTest/test_siracusa_tiled_config.py` | CI configuration: model registry, L1 sizes, per-model overrides |
| `DeeployTest/Tests/Models/Training/` | Test artifacts for all supported models |

---

## License

Apache 2.0 — see `LICENSES/` and individual file headers.
