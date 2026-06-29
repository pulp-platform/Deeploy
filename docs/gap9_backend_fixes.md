# GAP9 backend notes: `-O3` hot kernels & tiling/DMA table placement

Two GAP9 notes: one code change (`-O3` on the hot kernels) and one informational
note on *where the per-tile control tables live* and the L1 it buys. Both are about
the same theme — the GAP9 cluster runs on a small, manually-managed L1 (TCDM, 128 KB),
so what sits in L1 vs L2 dominates both speed and whether a net fits.

---

## 1. `-O3` on the hot forward/backward kernels

**File:** `TargetLibraries/GAP9/CMakeLists.txt`

**Problem.** The GAP9 SDK compiles the kernel sources at `-Os` (size) by default.
The conv / depthwise-conv / Gemm kernels (and their backward counterparts) dominate
GAP9 cycles, so `-Os` leaves the hottest code slow.

**Fix.** Compile just those translation units at `-O3` via
`set_source_files_properties(... PROPERTIES COMPILE_OPTIONS "-O3")`, **appended last**
so it wins over the SDK's `-Os` on the same files. Everything else stays at `-Os`.

**Why it helps.** `-O3` turns on the RISC-V (XpulpV2) **hardware loops** + unrolling
on the kernels' tight inner loops. Measured on a forward conv: the `-O3` object has
**18 `lp.setup`** hardware-loop instructions vs **0** at `-Os`, at the cost of ~+50 %
`.text` on those few files. The cycle win is kernel-dependent — modest on a
memory-bound generic conv, but the compute-bound matmul / conv-grad kernels (the bulk
of training cycles) speed up the most.

**Takeaway.** Per-file `-O3` on the few hot kernels is a large, cheap latency win;
ordering matters because the last `COMPILE_OPTIONS` wins.

---

## 2. Tiling + DMA descriptor tables: emit `const`, and they live in L2

**File:** `Deeploy/Targets/GAP9/Templates/AllocateTemplate.py`

**Code change.** The GAP9 global-init templates emitted the compile-time-initialized
arrays as `static PI_L2 … = {…}`. These include the per-tile control tables — the
tiling metadata and the DMA descriptor tables — which are **read-only lookup tables**
(the cluster controller only reads them to drive the tiling loop and program DMAs).
Emit them `static const PI_L2 … = {…}`. Correct (they are never written), it lets the
toolchain treat them as immutable read-only data, and it documents intent. Verified:
all 228 tables of a model go `const`, builds with **0** const-discard, runs identically.

**Why they live in L2 (and what L1 that saves).** Both table kinds are `PI_L2`
(→ `.data` in L2, **not** L1):
- **tiling metadata** — `DeeployNetwork_TILING_CODEGEN_*` arrays: `numTiles`,
  per-tensor `cumByteOffset`, tile geometry / transfer lengths.
- **DMA descriptor tables** — the `…_cmd[]` arrays holding each tile's L3/L2 transfer
  addresses, read when programming the L3↔L2 / L2↔L1 DMA.

Because they are `PI_L2`, they cost the L1 tile arena **nothing**. How much L1 that
keeps free grows with the net (measured, FP32 training):

| model | tiling metadata | DMA descriptor tables | **L1 kept free** |
|---|---|---|---|
| Autoencoder | 283 B | 0 B (no L3) | 283 B |
| DSCNN | 473 B | 24 B | 497 B |
| ResNet8 | 1733 B | 816 B | **2.5 KB** |
| MobileNetV1 | 5341 B | 1792 B | **7.0 KB** |
| CCT | 10452 B | 1432 B | **11.6 KB** |

The DMA tables are a real fraction (ResNet8 32 %, MNv1 25 %). For the L1-tight nets
this matters: CCT's L1 is nearly full (`arena 122000 + cc_stack 8192 = 130192 /
131072`), so the 11.6 KB the L2 tables keep out of L1 is headroom it cannot spare.

**What does keeping them in L2 cost?** Almost nothing — the cluster only *reads* a
few hundred bytes … ~12 KB of metadata, spread across all tiles. Measured by
relocating Autoencoder's tables `PI_L2 → PI_CL_L1` (into L1) and re-running:

| Autoencoder tables in… | train cycles |
|---|---|
| **L2** (default) | 3,111,938 |
| L1 | 3,093,012 |

→ moving them to L1 is only **0.6 % faster**, i.e. the L2-access cost is in the noise.

**Takeaway.** Read-mostly control tables belong in L2 — they free up to ~12 KB of L1
for the tile arena at ~0 cycle cost. (Contrast with hot, write-every-call data such as
the per-core stacks, which belong in L1.) Budget L1 with the tables already *out* of
it.
