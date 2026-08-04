<!--
SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna

SPDX-License-Identifier: Apache-2.0
-->

# NE16 on GAP9: Where the Peak Is, and How We Got Close To It

This note collects (1) the primary sources for the NE16 accelerator, (2) what its peak
throughput actually is and what you must do to approach it, and (3) the changes we made in
Deeploy to take MobileNetV1 from **9.43 to 13.96 MAC/cycle**, past the **10.3 MAC/cycle** the
GAP9 SDK reaches on a comparable model.

Every number below was measured on GVSoC (`gap9.evk`) through the Deeploy CI configuration.
Where a claim comes from reading source rather than from a measurement, it says so.

---

## 1. Sources

### 1.1 The accelerator itself

| What | Where | Notes |
|---|---|---|
| **NE16 RTL + docs** | [`pulp-platform/ne16`](https://github.com/pulp-platform/ne16) | The official repository. Maintained by Francesco Conti (University of Bologna / GreenWaves Technologies). |
| **RBE** (predecessor) | [`pulp-platform/rbe`](https://github.com/pulp-platform/rbe) | Reconfigurable Binary Engine, by Gianna Paulin and Francesco Conti. NE16 derives from it. |
| **pulp-nnx** (driver / HAL) | [`pulp-platform/pulp-nnx`](https://github.com/pulp-platform/pulp-nnx) | The task-descriptor layer Deeploy generates calls into (`ne16_task.c`, subtile counters, strides). |

There is **no paper describing the NE16 microarchitecture itself**. It is a productised IP in
GAP9, and its details live in the repository and in the SDK. The publication the NE16 README
cites is the ancestor design:

> F. Conti, P. D. Schiavone, L. Benini, *"XNOR Neural Engine: A Hardware Accelerator IP for
> 21.6-fJ/op Binary Neural Network Inference"*, IEEE Transactions on Computer-Aided Design of
> Integrated Circuits and Systems, vol. 37, no. 11, 2018, pp. 2940–2951.

Systems papers that use NE16 and report end-to-end numbers (useful for calibration, not for
microarchitecture): *Flexible and Fully Quantized Ultra-Lightweight TinyissimoYOLO*
([arXiv:2307.05999](https://arxiv.org/pdf/2307.05999)), *GAP9Shield*
([arXiv:2407.13706](https://arxiv.org/html/2407.13706v1)).

### 1.2 The behavioural model — the most useful source in practice

GVSoC ships a cycle-level C++ model of NE16. When the documentation is ambiguous, **this is the
ground truth**, and it is readable:

```
$GAP_SDK/gvsoc/gvsoc_gap/gap/ne16/src/
├── ne16_regfile.cpp     ← CONFIG0 bit decode. Authoritative bit map.
├── ne16_matrixvec.cpp   ← the MAC array itself
├── ne16_normquant.cpp   ← scale/bias/shift. Note: plain `>> shift`, no rounding term.
├── ne16_streamout.cpp   ← output saturation (signed [-128,127] / unsigned [0,255])
└── ne16_load.cpp        ← input fetch
```

Two examples of questions we answered by reading it rather than guessing:

* **Does NE16 round or truncate in requantisation?** `ne16_normquant.cpp` does
  `accum32[i] >> shift` with no rounding term added first. A golden model copied from the
  *software* `RequantShift_s8.c` kernel (which rounds half-up) will disagree with hardware by
  up to 1 LSB.
* **Which CONFIG0 bits exist?** `ne16_regfile.cpp` decodes `[4]` outquant, `[6:5]` filter mode,
  `[7]` linear, `[8]` strided-2x2, `[11:9]` **reserved**, `[13:12]` norm bits, `[14]` streamin,
  `[15]` weight-offset (marked *"FIXME not implemented"*), `[20:16]` quant shift, `[22:21]`
  quant bits, `[23]` quant-norect, `[24]` norm shift, `[25]` norm bias. Anything you set
  outside that map is a no-op.

Register bit names are in `$GAP_SDK/tools/autotiler_v3/CNN_Libraries_HWPE/hal_ne16.h`
(`NE16_REG_CONFIG 0x5c`, `NE16_SHIFT_*`).

### 1.3 The analytical performance model — read this before optimising

[`dory/Hardware_targets/PULP/GAP9_NE16/Tiler/Ne16PerfModel.py`](https://github.com/pulp-platform/dory/blob/master/dory/Hardware_targets/PULP/GAP9_NE16/Tiler/Ne16PerfModel.py)

DORY's model decomposes one NE16 job into pipeline stages and gives a closed-form cycle count:

```
total = n_spatial × [ n_out_body × iteration_latency(k_out_body) + iteration_latency(k_out_rem) ]

FIFO_LATENCY = 6      SHIFTER_COUNT = 4     ADDER_COUNT = 8
MULTIPLIER_COUNT = 4  MEMORY_THROUGHPUT = 256 bit/cycle
```

The term that matters is **`k_out_rem`**: every job pays a fixed setup cost, and a tile that
does not fill the array pays it for partial work. Utilisation is the ratio of real MACs to
`max_ops`, which the model derives from the *padded-up* tile counts.

### 1.4 What the SDK's own tiler does

`$GAP_SDK/tools/autotiler_v3/CNN_Generators_NE16/CNN_Generators_NE16.c` — worth reading because
it encodes GreenWaves' own answer to "how do I keep NE16 busy":

| Line | Code | Meaning |
|---|---|---|
| `669`, `1115` | `OutTileCons = CannotTileChannels ? OutFeat : 32` | prefer output-channel tiles that are a multiple of **32** |
| `687`, `1185` | `InTileCons = Mode16 ? 8 : 16` | input-channel tiles multiple of **16** for 8-bit |
| `1132` | `Fcx==3 && Fcy==3 && (s==1 \|\| s==2)` → `O_NE16_3X3` | 3x3 **stride 2 is native**, not a fallback |
| `1186` | `AllowActFusion && ActOper != KOP_NONE` | conv + activation fused into one kernel |

Note `CannotTileChannels ? OutFeat : 32` — the alignment is a *preference that degrades*, never
a hard constraint. Layers with fewer than 32 output channels must still be tileable.

---

## 2. Peak performance, and why you will not reach it

### 2.1 The number

The array is **9 × 9 engines × 16 input channels**, each engine performing one binary
multiply-accumulate per cycle:

```
9 × 9 × 16 = 1296 binary MAC/cycle
```

NE16 is bit-serial in the weights: an 8-bit weight takes 8 passes. So for the usual
8-bit × 8-bit case:

```
1296 / 8 = 162 MAC/cycle          ← theoretical peak, 8-bit weights
```

At GAP9's 370 MHz that is ~60 GMAC/s; the commonly quoted **32.2 GMAC/s** figure corresponds to
sustained real-workload throughput, not the array bound.

### 2.2 The three alignment rules

Derived from the array geometry, and independently confirmed by what the SDK's tiler enforces:

| Rule | Why |
|---|---|
| `Ci % 16 == 0` | `TP_IN = 16`: each engine consumes 16 **contiguous** input channels per cycle. This also forces a **channels-last (HWC)** layout — it is not a preference, it is how the datapath is fed. |
| `Co % 32 == 0` | `TP_OUT = 32` output channels retire per pass. A tile of 3 or 56 output channels wastes most of the output lanes for the *whole* tile. |
| `Ho % 3 == 0`, `Wo % 3 == 0` | the 9 columns produce a 3×3 output patch per pass. |

Plus: stride 1 or 2 only, `qw = 8` for the 162 figure (lower weight precision scales linearly —
4-bit weights double it).

### 2.3 What actually happens on a real network

Measured, MLPerf Tiny VisualWakeWords (MobileNetV1 0.25×, 96×96, 7,489,664 MAC):

| Configuration | MAC/cycle | % of 162 |
|---|---|---|
| Single-layer dense conv, 64→64 ch, 32×32 | **74.33** | 45.9 % |
| Full network, this work | **13.96** | 8.6 % |
| Full network, GAP9 SDK (comparable size) | 10.3 | 6.4 % |

**A whole network runs at roughly a fifth of what a single well-shaped layer achieves, and both
are far from 162.** The reasons are structural, not fixable by tuning:

* MobileNetV1 0.25× has layers with **8, 16, 32** channels against `TP_IN=16` / `TP_OUT=32`.
  The first layer uses 8 of 32 output lanes. No tiler can fix a model that is narrower than the
  datapath.
* Depthwise layers have one input channel per output channel by construction, so the 16-wide
  input dimension is inherently underfilled.
* Everything that is not a MAC — layout conversion, tile DMA, job setup — is pure overhead.

The practical consequence, and the point worth making to anyone tuning this: **beyond the
alignment rules, the remaining wins are in removing non-compute work, not in feeding the array
better.** Our 1.48× came entirely from the former — the NE16 dispatch count did not change at
all.

---

## 3. What we changed, and why

Four changes, in descending order of impact. All were verified bit-exact against the
pre-existing golden outputs; the NE16 CI jobs (`kernels`/`models` × `singlebuffer`/`doublebuffer`,
L2) pass.

### 3.1 Fold the redundant layout transposes — 1.48× on the full network

**Symptom.** The generated `Network.c` for VisualWakeWords contained **26 transpose passes for
27 convolutions**: a `_pre_transpose` / `_transpose` pair wrapped around essentially every
convolution, 232 cluster forks, each half carrying its own tiling loop and L2↔L1 DMA round trip.

**Cause.** ONNX is NCHW; NE16 requires HWC (§2.2). `PULPNCHWtoNHWCPass` inserts the conversions,
and `PULPOpenDeployer` already ends its lowering chain with the clean-up that folds them:

```python
PULPNCHWtoNHWCPass(...)   TransposeSplitPass()   RQAddTransposeSquashPass()
TransposeSplitPass()      TransposeMergePass()   TransposeConstOptPass()
ReshapeConstOptPass()     TransposeNoPermOptPass()
```

But `NE16Deployer` then does `self.loweringOptimizer.passes += [...]`, so
`NE16OptimizationPass` — which inserts layout transposes of its own via `_appendTranspose` —
runs **after** that clean-up. Its transposes were never folded. Two consecutive NE16 convs were
therefore separated by a `HWC→CHW` followed by a `CHW→HWC`: an identity pair that survived all
the way into generated code.

**Fix.** Re-run the same clean-up chain after `NE16OptimizationPass` (4 lines,
`Targets/NE16/Deployer.py`).

| | before | after |
|---|---|---|
| transpose cluster forks | 232 | **16** |
| tiling loops | 114 | **62** |
| NE16 dispatches | 56 | **56** (compute untouched) |
| cycles | 794,080 | **536,521** |
| MAC/cycle | 9.43 | **13.96** |

The general lesson: when a subclass appends passes with `+=`, whatever the base class ran as a
*final* clean-up is no longer final.

### 3.2 Pin the weight tile's encoded tail — a 2 KB buffer overrun

**Symptom.** Many tiled dense configurations produced wrong results: 2047 wrong outputs for
`64/64 @32×32`, 8097 for `16/16 @64×64`, 10121 for `4/4 @128×128`. Error counts scaled with the
tile count, were invariant to the `--l1` value and to arena placement, and untiled runs were
always correct.

**Diagnosis.** The wrong outputs formed **two contiguous 1024-byte runs exactly 32768 bytes
apart** — one output tile's worth, same offset within each tile. That shape says *memory
overwrite*, not arithmetic. Dumping the generated L1 layout:

```
data_in  @      0   size 65536      →      0 – 65535
data_out @  65536   size 32768      →  65536 – 98303
weight   @  98304   size 18432      →  98304 – 116735   ← overruns
mul      @ 114688   size   128
add      @ 114816   size   128
```

`serializeTilingSolution` always emits the weight tile as `(CSize,) + weightShape[1:]` — only the
output-channel dimension is tiled, the NE16-encoded tail is always moved whole. But
`addGeometricalConstraint` only pinned `weightOutChannelVar`, leaving `(cinMajor, bits,
H*W*cinMinorBytes) = (4, 8, 18)` free. The solver shrank the last dimension to 16 and reserved
`32×4×8×16 = 16384` B while the DMA writes `32×4×8×18 = 18432` B. The extra 2048 B landed on the
requantisation `mul`/`add` parameters — and exactly 2048 B of output came out wrong.

**Fix.** Constrain the three tail dimensions to their maximum (3 lines,
`NE16DenseConstraint.addGeometricalConstraint`). All three failing shapes go to **0 errors**; the
control shape is cycle-identical.

### 3.3 Prefer output-channel tiles that are a multiple of 32

Nothing expressed §2.2's `Co % 32` rule to the tiler, and the solver was free to pick whatever
fit — `Ko = 3` and `Ko = 56` were both observed in generated code. Added as a
`PerformanceHint` (not a hard constraint, matching the SDK's `CannotTileChannels` degradation)
to the dense, depthwise and pointwise constraints. This also makes shapes tileable that
previously could not be tiled at all — e.g. `8/8 @96×96`, which now runs bit-exact.

### 3.4 Make `SLAVESTACKSIZE` overridable

The cluster slave stacks were pinned at 3800 B/core by an unconditional `#define`, so ~32 KB of
the 128 KB L1 was gone before the tiling arena started — while the tiler was still told it had
the full `--l1` budget. Values above ~98000 either failed to allocate or, worse, produced a tile
layout that overran L1 and showed up only as a DMA out-of-bound trace at run time.

Two edits are needed and **either one alone is a silent no-op**: the GAP9 `CMakeLists.txt` must
turn the `-D` cache variable into a compile definition, *and* the `#define` in `deeploytest.c`
must be wrapped in `#ifndef` or it shadows the command-line one. Arena goes from 98,176 to
~121,000 B.

Caveat: this is a knob, not a free win. Slave stacks below ~1024 B crash the cluster kernels
that VisualWakeWords still runs (`Invalid fetch request (addr: 0x0)` — a clobbered return
address), and on this model the extra L1 buys nothing, because single-buffer is not L1-bound:
`--l1` 128000→131000 × stack 3800/1280/1024 all give an identical 794,080 cycles.

---

## 4. Result

| Configuration | cycles | MAC/cycle | vs SDK |
|---|---|---|---|
| Baseline (before this work) | 794,080 | 9.43 | 92 % |
| **Single-buffer, L1 128000** | **536,521** | **13.96** | **136 %** |
| Double-buffer, L1 100000 | 567,140 | 13.21 | 128 % |
| GAP9 SDK, comparable model size | — | 10.3 | 100 % |

Single-buffer is the better configuration here: double-buffering must fit two of every tile in
L1, so it tiles more finely, and the extra splits cost more than the overlapped DMA saves.

---

## 5. Reproducing

```bash
source $GAP_SDK/.gap9-venv/bin/activate
source $GAP_SDK/configs/gap9_evk_audio.sh          # NOT gap9_v2.sh — the target must
                                                    # match --target=gap9.evk or the chip
                                                    # never boots and gvsoc hangs silently
export GVSOC_INSTALL_DIR=$GAP_SDK/install/workstation
export GAP_RISCV_GCC_TOOLCHAIN=/path/to/gcc/gap9
export CCACHE_DIR=<writable>

cd DeeployTest
pytest test_platforms.py -v -s -m "gap9_w_ne16_tiled and models and singlebuffer and l2"
```

Use `pytest` with markers, never the single-kernel runner with hand-picked flags: the per-model
overrides (L1 budget, `gen_args`) live in `test_gap9_ne16_tiled_config.py` and
`test_platforms.py`. Bypassing them cost us a 44 % discrepancy on the same nominal `--l1`,
because the direct runner does not accept `--enableStrides` and the stride-2 layers silently
fell back to the cluster.

Also: **wipe `DeeployTest/TEST_GAP9_W_NE16` between experiments.** CMake uses `file(GLOB)` at
configure time, and a stale `build_master` will happily re-run a previous binary — which
produced four consecutive wrong diagnoses in the course of this work.

---

## 6. Open items

* **Signed activations are not supported.** `ConvTemplate.getConf0` sets `conf0 |= 1 << 26` for
  `input_signed`, but bit 26 is not part of CONFIG0 (§1.2) — it is a no-op, so int8 activations
  are consumed as uint8. NE16 has no signed-input mode; the correct approach is an offset
  correction (+128 on the input, `-128 × Σweights` folded into the bias). Post-ReLU networks
  such as MobileNet are unaffected, which is why this has gone unnoticed.
* **Activation fusion.** The SDK fuses conv + activation into a single kernel
  (`CNN_Generators_NE16.c:1186`); Deeploy runs them as separate passes. This is the most likely
  source of the remaining gap on networks where the transposes are already folded.
* **Narrow layers.** Nothing in the tiler exploits the fact that a `Ci = 8` layer wastes half
  the input datapath. The SDK does not appear to either, but it bounds what either can achieve.
