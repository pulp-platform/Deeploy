.. SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
..
.. SPDX-License-Identifier: Apache-2.0

Microbenchmark
==============

Pass ``--profileMicrobenchmark`` to any PULPOpen runner (``testMVP.py``, ``generateNetwork.py``, ``deeployRunner_*.py``) to wrap each layer in ``RunNetwork`` with PULP performance counters. Off by default; zero overhead when unused.

The flag flows through :py:attr:`Deeploy.DeeployTypes.CodeGenVerbosity.microbenchmarkProfiling` into :py:class:`Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPMicrobenchmark.PULPMicrobenchmark`, which is registered last in the PULPOpen ``ForkTransformer`` and ``ClusterTransformer`` chains so it covers the full per-layer body (tiling, DMA, memory management). The C-side helpers live in ``TargetLibraries/PULPOpen/inc/perf_utils.h``.

Each layer prints one block on ``core 0``:

.. code-block:: text

    === Performance Statistics: Add_0 ===
    Cycles:                    1442
    Instructions:               149
    IPC:                      0.103
    Loads / Stores / Branches / Taken Branches / RVC
    Load Stalls / Jump Stalls / I-cache Misses / TCDM Contentions
    External Loads / Stores and their cycle counts

External-memory and TCDM-contention counters are zero when the wrapped region has no L2/L3 traffic or no bank conflicts (e.g. small untiled kernels that fit in L1). Some events may not be modelled by GVSoC — verify on a tiled test before assuming a counter is broken.
