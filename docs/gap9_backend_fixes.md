# GAP9 backend fixes & memory-tuning knobs

Short notes on a set of GAP9 backend changes: what each one fixes, why, and where
it lives. They share a theme — **the GAP9 cluster runs on a small, manually-managed
L1 (TCDM), and HyperRAM/L3 is not CPU-addressable** — so most bugs here are about
*where memory lives* and *how DMA waits complete*. GVSoC models memory more
forgivingly than real silicon (flat HyperRAM, generous timing), so several of these
pass in simulation and only fault on the board.

## GAP9 memory model (background)

| Level | Address window | Who can touch it directly |
|---|---|---|
| L1 / TCDM (128 KB) | `0x1000_0000–0x1004_0000` | cluster cores (PEs) + CC |
| L2 (1.5 MB) | `0x1C00_0000–0x1C20_0000` | FC + cluster (shared) |
| L3 / HyperRAM | `cl_ram_malloc` handles | **DMA only** — not CPU-addressable |

The cluster has 8 worker PEs + a cluster-controller (CC / master) core. The CC
stack is carved from the **bottom of L1**; PE slave stacks are also L1 by default.
Everything competes with the Deeploy tile arena for those 128 KB.

---

## 1. L3-aware input/output in the board test harness

**File:** `DeeployTest/Platforms/GAP9/src/deeploytest.c`

**Problem.** The harness classified buffers by raw address thresholds
(`ptr >= 0x10000000` for inputs, `< 0x10000000` for outputs). HyperRAM/L3 buffers
(`cl_ram_malloc`) are *also* `>= 0x10000000`, but HyperRAM is **not CPU-addressable**.
So for `--defaultMemLevel L3` tests, `main()` did a raw `memcpy` into an L3 input
pointer and CPU-dereferenced L3 output pointers — an `Invalid fetch` fault on the
board, right after init, before any results print. GVSoC models HyperRAM as flat
RAM, so it passed there and masked the bug.

**Fix.** Add `IS_L1` / `IS_L2` on-chip-window macros and use them: on-chip inputs
are `memcpy`'d, **L3 inputs are already loaded from the readfs hex inside
`InitNetwork`** (their `testInputVector` entry is `NULL`) so they're skipped, and
**L3 outputs are `ram_read` into an L2 scratch** before the compare.

**Takeaway.** Never CPU-`memcpy`/deref an L3 pointer on GAP9 — gate on the real
on-chip windows, not a single `>= 0x10000000` threshold.

---

## 2. Split the L3 tiling DMA — blocking for single-buffer, async for double-buffer

**Files:** `Deeploy/Targets/PULPOpen/CodeTransformationPasses/PULPL3Tiling.py`,
`Deeploy/Targets/GAP9/Bindings.py`, `Deeploy/Targets/GAP9/DMA/L3Dma.py`

**Problem.** L3↔L2 tiling used one DMA backend for both single- (SB) and
double-buffering (DB). Async DMA only helps DB, where it overlaps the *next* tile's
prefetch with compute. SB waits on each tile before computing, so async there buys
nothing but adds risk: strided 2D L3 transfers (`pi_cl_ram_copy_2d`) can corrupt
under deferred waits.

**Fix.** `PULPL3Tiling` gains an optional `dbDma` (defaults to `dma`, so it is
backward compatible). GAP9 binds **SB → blocking** `gap9L3DmaHack`, **DB → async**
`GAP9L3Dma`. Also: reset the L3 future's `.size` to 0 after `pi_cl_ram_copy_wait`
(so a completed future is never waited on twice) and cast `${ext}` to `uint32_t`
in the 2D transfer.

**Takeaway.** Async DMA is a double-buffering optimization; don't pay its hazards
on the single-buffering path.

---

## 3. `-O3` on the hot forward kernels

**File:** `TargetLibraries/GAP9/CMakeLists.txt`

**Problem.** The SDK compiles kernels at `-Os` by default; the conv / depthwise-conv
/ Gemm kernels dominate GAP9 inference cycles and are left slow.

**Fix.** Compile `Convolution_fp32.c`, `DWConvolution_fp32.c`, `Gemm.c` at `-O3`
via `set_source_files_properties(... COMPILE_OPTIONS "-O3")`, **appended last** so
it wins over the SDK's `-Os` on the same translation units.

**Takeaway.** Per-file `-O3` on the few hot kernels is a large, cheap latency win;
ordering matters because the last `COMPILE_OPTIONS` wins.

---

## 4. L1-memory tuning knobs (documented in the example)

**Files:** `DeeployTest/Platforms/GAP9/src/deeploytest.c`,
`DeeployTest/Platforms/GAP9/CMakeLists.txt`,
`DeeployTest/Platforms/GAP9/sdk_gvsoc.config`

Three independent ways to free L1 TCDM for the tile arena so conv-heavy nets fit.
The example demonstrates all three with comments.

- **Knob A — slave (PE) stacks → L2.** The SDK `pi_cl_l1_malloc`'s the PE stacks
  only when `task->stacks == NULL`. Hand it a static buffer (`SET_SLAVE_STACK`,
  a `.bss` array → L2) and it skips that L1 allocation, freeing ~30 KB of L1
  (8 cores × `SLAVESTACKSIZE`). **But this is *not* free:** the PE stacks are
  written hot (every call/spill), so moving them to L2 costs **+22…63 % train
  cycles** (measured: AE +38 %, DSCNN +34 %, ResNet8 +24 %, MNv1 +23 %).
  **Prefer Knob B:** the FP32 kernels keep working sets in explicit L1 buffers, so
  the real per-core stack need is <256 B vs the 3800 B default — shrink
  `SLAVESTACKSIZE` to ~512 B and keep the stacks **in L1** (`stacks == NULL`),
  which reclaims ~26 KB of the 30 KB *without* the L2 penalty. Use L2 stacks
  (this knob) only when the L1 is so full the arena needs every byte (e.g. CCT,
  whose big gemms also make the L1-vs-L2 stack difference negligible anyway).
- **Knob B — shrink the SDK's L1 slave stacks.** `CONFIG_CL_SLAVE_CORE_STACK_SIZE`
  in the sdk `.config` (alternative to Knob A; use one or the other).
- **Knob C — cluster-controller stack size.** The CC stack grows down from the L1
  base; the SDK default (`0x800` = 2 KB) is too small for deep tiling call chains
  and overflows below the base (silent clobber / invalid write). Set
  `conf.cc_stack_size`, overridable from the build with **`-DCC_STACK_SIZE=<bytes>`**
  (new CMake option). Example (CI): `cmake … -DCC_STACK_SIZE=8192`.

**Takeaway.** The CC stack and PE stacks are *invisible to the tiler and the ELF*
(carved at runtime), yet they share L1 with the arena — budget for them explicitly.

---

## 5. Emit cluster fork/closure argument structs as `static`

**File:** `Deeploy/CommonExtensions/CodeTransformationPasses/MemoryAllocation.py`
(`ArgumentStructGeneration`)

**Problem.** The per-node tiling argument structs were stack-locals in the
dispatching function. The cluster-fork runtime writes its descriptor near the top
of the CC/master stack and can clobber a stack-local arg struct **before the forked
cores read it** — a GAP9 cluster-fork crash (seen on MobileNetV1).

**Fix.** Declare the struct `static` (off-stack) and assign separately:
`static T name; name = (T){…};`. Static storage keeps it stable for the lifetime
of the forked call. Generic codegen, benign on other targets.

**Takeaway.** Anything handed to a cluster fork must outlive the CC stack frame —
keep it off the stack.

---

## 6. Per-tensor waiting strategy for the cluster (mchan) DMA

**File:** `Deeploy/Targets/GAP9/DMA/MchanDma.py`

**Problem.** GAP9 mchan allocates a fresh channel on every descriptor enqueue.
`DirectionWaitingStrategy` shares **one** future (`one mchan_transfer_get_id`)
across all same-direction tensors of a tile. A tile with >1 input (e.g. optimizer
weight + grad) then emits one `get_id` but multiple pushes → the extra transfers
run on channels that are never waited or freed → `mchan_transfer_wait()` hangs
forever.

**Fix.** Use `PerTensorWaitingStrategy`: each tensor gets its own
`get_id : push : wait : free`, matching the mchan hardware contract.

**Takeaway.** Match the DMA waiting strategy to the hardware's channel model —
mchan is one-channel-per-transfer, so wait per tensor, not per direction.

---

## 7. Build-time memory gate

**Files:** `DeeployTest/testUtils/gap9_memcheck.py`,
`DeeployTest/testUtils/core/execution.py`

**Problem.** L1/L2 over-subscription on GAP9 surfaces as a multi-minute GVSoC hang
(`os_evt_release`) or a wild-pointer crash far from the cause — slow and opaque.
The tiler does not model the CC master stack, PE slave stacks, or the promoted
pool, so it can't catch it.

**Fix.** `gap9_memcheck.py` models every consumer of L1 and L2 (tile arena, CC
stack from `cc_stack_size`, PE slave stacks, ELF sections, promoted pool) and
scans `InitNetwork` for the `pi_l2_malloc`-after-`cl_ram_malloc` alloc-order race.
`run_complete_test` runs it after the build and before the simulation (GAP9 only),
so over-subscription fails in seconds with the exact knob to turn. Bypass with
`DEPLOY_SKIP_MEMCHECK=1`.

**Takeaway.** Validate the full L1/L2 budget at build time — the stacks and pools
the tiler ignores are exactly what overflow.

---

## 8. Tiling + DMA descriptor tables live in L2 — a (near-free) L1 win

**Where:** the tiler emits, per tiled node, two kinds of control table, both
`static PI_L2` (→ `.data` in L2):
- **tiling metadata** — `DeeployNetwork_TILING_CODEGEN_*` arrays: `numTiles`,
  per-tensor `cumByteOffset`, tile geometry / transfer lengths.
- **DMA descriptor tables** — the `…_cmd[]` arrays holding each tile's L3/L2
  transfer addresses (consumed when programming the L3↔L2 / L2↔L1 DMA).

Because they are `PI_L2`, they sit in L2, **not** L1 — so they cost the tile arena
nothing. How much L1 that saves grows with the net (measured, FP32 training):

| model | tiling metadata | DMA descriptor tables | **total L1 saved** |
|---|---|---|---|
| Autoencoder | 283 B | 0 B (no L3) | 283 B |
| DSCNN | 473 B | 24 B | 497 B |
| ResNet8 | 1733 B | 816 B | **2.5 KB** |
| MobileNetV1 | 5341 B | 1792 B | **7.0 KB** |
| CCT | 10452 B | 1432 B | **11.6 KB** |

The DMA tables are a real fraction (ResNet8 32 %, MNv1 25 %). For the L1-starved
nets this is load-bearing: **CCT's L1 is full** (`arena 122000 + cc_stack 8192 =
130192 / 131072`), so the 11.6 KB the L2 tables keep out of L1 is exactly the
headroom it cannot spare.

**What does keeping them in L2 cost?** Almost nothing — the cluster only *reads*
a few hundred to ~12 KB of metadata, spread across all tiles. Measured by
relocating Autoencoder's tables `PI_L2 → PI_CL_L1` (into L1) and re-running:

| Autoencoder tables in… | train cycles |
|---|---|
| **L2** (default) | 3,111,938 |
| L1 | 3,093,012 |

→ moving them to L1 is only **0.6 % faster**, i.e. the L2-access cost is in the
noise. So the tables belong in L2: they free up to 11.6 KB of L1 at ~0 cycle cost.

**Takeaway.** Tables in L2 are the *opposite* trade from PE stacks in L2 (Knob A
above): tables are read-mostly and tiny relative to their L1 value, so L2 is the
right home; stacks are written hot every call, so L2 is expensive for them. Place
read-mostly control data in L2, keep hot working/stack data in L1.

---

*All changes verified on GVSoC with `MatMul --defaultMemLevel L3` (`Errors: 0 out
of 256`). On-chip (L1/L2) behaviour is unchanged; only the L3 / stack / DMA paths
differ.*
