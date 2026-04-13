.. SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
..
.. SPDX-License-Identifier: Apache-2.0

Per-Layer Microbenchmarking on PULPOpen
=======================================

Deeploy can wrap each layer in the generated ``RunNetwork`` with PULP performance-counter instrumentation, producing per-layer reports of cycles, instructions, stalls, instruction-cache misses, branch behaviour, and external/TCDM memory traffic. This is intended for profiling individual layers of a deployed network on real hardware or in GVSoC, without modifying any kernel source.

The instrumentation is **off by default** and adds zero overhead unless explicitly enabled.

Enabling
--------

Pass ``--profileMicrobenchmark`` to any of the runner entry points:

.. code-block:: bash

    python testMVP.py        ... --profileMicrobenchmark
    python generateNetwork.py ... --profileMicrobenchmark
    python deeployRunner_siracusa.py -t Tests/Kernels/FP32/Add/Regular --profileMicrobenchmark

The flag flows through :py:attr:`Deeploy.DeeployTypes.CodeGenVerbosity.microbenchmarkProfiling`
into the :py:class:`Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPMicrobenchmark.PULPMicrobenchmark`
code-transformation pass, which is registered at the outermost position of the PULPOpen
``ForkTransformer`` and ``ClusterTransformer`` chains. Because it runs last, the wrapped region
covers the full per-layer body, including all tiling, DMA, and memory-management code.

Output Format
-------------

Each layer emits one block of statistics on ``core 0``:

.. code-block:: text

    === Performance Statistics: Add_0 ===
    Cycles:                    1442
    Instructions:               149
    IPC:                      0.103

    --- Instruction Mix ---
    Loads:                       24 (16.11%)
    Stores:                      27 (18.12%)
    Branches:                     5 (3.36%)
    Taken Branches:               2 (40.00%)
    Compressed (RVC):             0 (0.00%)

    --- Stalls & Hazards ---
    Load Stalls:                  0
    Jump Stalls:                  0
    I-cache Misses:             724
    TCDM Contentions:             0

    --- Memory Hierarchy ---
    External Loads:               0 (0.00%)
    External Stores:              0 (0.00%)
    Ext Load Cycles:              0 (avg: 0.00)
    Ext Store Cycles:             0 (avg: 0.00)
    ========================================

Underlying Helpers
------------------

The C-side helpers live in ``TargetLibraries/PULPOpen/inc/perf_utils.h`` and are included by
default in PULPOpen builds via ``Platform.py``. The pass injects:

- ``perf_bench_init()`` / ``perf_bench_start()`` / ``perf_bench_read(&start)`` before the layer body
- ``perf_bench_stop()`` / ``perf_bench_read(&end)`` / ``perf_bench_diff(&total, &end, &start)`` /
  ``perf_bench_print("<layer>", &total)`` after it

All counters listed in ``perf_stats_t`` are configured at once in ``pi_perf_conf``, so a single
wrap captures the full event set.

Notes & Caveats
---------------

- **External memory counters** (``LD_EXT``, ``ST_EXT``, ``LD_EXT_CYC``, ``ST_EXT_CYC``) only show
  non-zero values when the wrapped region performs L2/L3 traffic. Untiled tests that fit in L1/TCDM
  will report zero.
- **TCDM contention** depends on the access pattern — regular, bank-friendly kernels (e.g. element-wise
  Add) can legitimately report zero contention even with all 8 cores active.
- Some events may not be modelled by GVSoC; verify on a tiled test (e.g. Siracusa-tiled GEMM) before
  concluding a counter is broken.
- Output is printed by ``core 0`` only to keep logs readable.
