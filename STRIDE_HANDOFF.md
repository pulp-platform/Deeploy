# NE16: what "the stride bug" actually was, and where the real headroom is
(2026-08-04, second pass)

## 1. Stride is NOT broken, and NOT the bottleneck. Premise disproven.

VisualWakeWords (MobileNetV1) generated code:
    NE16 dispatches (ne16_nnx_dispatch):  56
    cluster conv kernels (pulp_nn_conv*):  0
All 27 conv passes -- including every stride-2 downsample -- already run on NE16.
There is no speedup available from "implementing stride"; it is already done.
pytest passes `--enableStrides` via gen_args; only the *direct runner* lacked
the CLI flag (now added, see uncommitted changes).

## 2. What StriddedPadded_2D_RQ actually exposes: signed inputs are unsupported

Error detail (this is what I should have read first):
    Expected: 127  Actual: -128  Diff: -1   x6 of 8

Expected is saturated +127, actual is saturated -128 -- the accumulator's SIGN
is wrong, not its addressing.

Input ranges across the NE16 test suite:
| test | input range | signed | result |
|---|---|---|---|
| StriddedPadded_2D_RQ | -128..124 | YES | 6/8 wrong |
| DW_2D_RQ | 0..255 | no | 0 errors |
| PW_2D_RQ/Regular_RQ | 0..254 | no | 0 errors |
| Dense_2D_RQ | 0..3 | no | 0 errors |

Every passing NE16 test is unsigned; the only signed one fails. NE16 has **no
signed-input config bit**: gvsoc `ne16_regfile.cpp:200-225` decodes CONFIG0 as
[4] outquant, [6:5] filter mode, [7] linear, [8] strided2x2, [11:9] RESERVED,
[13:12] norm bits, [14] streamin, [15] weight-offset (marked "FIXME not
implemented"), [20:16] quant shift, [22:21] quant bits, [23] quant norect,
[24] norm shift, [25] norm bias. **Bit 26 is not decoded at all** -- and
`ConvTemplate.getConf0` sets `conf0 |= 1 << 26` for `input_signed`. That is a
phantom bit; signedness never reaches the hardware, so int8 is consumed as
uint8. Bit 9 (`use_wmem` in Deeploy) also lands in the reserved [11:9] field
and deserves a separate look.

Correct approach for signed input on NE16 is an offset correction: shift the
input by +128 (making it unsigned) and subtract 128*sum(weights) through the
bias. Not implemented anywhere in Deeploy today.

MobileNet/VWW is unaffected: its activations are post-ReLU and unsigned.

## 3. The real gap vs the GAP9 SDK: per-layer layout transposes

VisualWakeWords generated code contains **26 transpose passes for 27 conv
passes** -- a `_pre_transpose` / `_transpose` pair wrapped around essentially
every convolution, 232 cluster-fork calls in total, plus their own L2<->L1 DMA
round trips. Pure data movement, zero MACs.

GAP9's AutoTiler converts the whole network to HWC **once at import**, so there
are no per-layer conversions at all. That is the structural difference behind
our 9.44 MAC/cycle vs the SDK's 10.3 on this model size.

Puzzle to start from: `Deeploy/Targets/NE16/Deployer.py:24` already sets
`default_channels_first = False`, so the graph should be channels-last globally
and these transposes should not exist. Find what re-introduces them (a topology
pass? a non-conv op that demands NCHW? the network's own ONNX?).

## Measurements to compare against (all gvsoc, VWW = MobileNetV1, 7.49 MMAC)
| config | cycles | MAC/cycle |
|---|---|---|
| singlebuffer @128000 (CI default, best) | 794,080 | 9.43-9.44 |
| double buffer @110000 + SLAVESTACKSIZE=1280 | 830,146 | 9.02 |
| double buffer @90000 (old CI default) | 860,577 | 8.70 |
| GAP9 SDK, closest model size | -- | 10.3 |

Single-buffer is NOT L1-bound: l1 128000..131000 x slave stack 3800/1280/1024
all give an identical 794,080 cycles. Memory tuning is exhausted.

## Uncommitted working-tree changes (all backed up)
- `Targets/NE16/Templates/ConvTemplate.py` (/tmp/ConvTemplate.bak)
  Stride-aware input extent in getCounters (dense + DW):
  `(height_out_border-1)*strideH + 3 - padding_bottom`. Collapses to the old
  `+2` when S=1, so stride-1 is bit-identical (verified). Correct in principle,
  still UNVALIDATED for S=2 -- needs a case whose *border* subtile has Ho>=2.
  (The conf0 `1 << 8` strided-mode bit was tried and REVERTED: it changed 10680
  -> 10598 cycles and zero errors, i.e. irrelevant to the failure.)
- `DeeployTest/deeployRunner_tiled_gap9_w_ne16.py` (/tmp/rn.bak) and
  `testUtils/deeployRunner.py` (/tmp/dr.bak): expose/forward `--enableStrides`.

## Already on PR #183 (pushed, CI models test passes 0 errors)
    d8bef5e9 test(NE16): raise VWW double-buffer L1 budget to 110KB
    4e88a10a perf(NE16): prefer output-channel tiles multiple of TP_OUT=32
    1c37e813 fix(GAP9): hoist L2->L1 tile-control tables to L2
    a47e812d fix(GAP9): make SLAVESTACKSIZE overridable from CMake

## Also open
per-tile boundary bug: error count scales with tile count, invariant to L1 and
arena placement. Localise by reducing failing output indices modulo the tile
output geometry.
