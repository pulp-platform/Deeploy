<!--
SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna

SPDX-License-Identifier: Apache-2.0
-->

# XDNA2 Execution Tracing

The XDNA2 backend supports optional AIE execution tracing.
When enabled, the generated MLIR includes trace configuration for both **core
events** (instruction execution, stalls, port activity) and **memory events**
(DMA start/finish/starvation).
After execution on the NPU, the raw trace data is parsed into a JSON file
that can be visualised in [Perfetto](https://ui.perfetto.dev/).

## Quick Start

From the `DeeployTest/` directory:

```bash
# Generate, build, run on the NPU, and parse the trace in one step:
python deeployRunner_xdna2.py -t Tests/Kernels/BF16/Add/Regular --trace

# With a custom trace buffer size (default: 8192 bytes):
python deeployRunner_xdna2.py -t Tests/Kernels/BF16/Add/Regular \
    --trace --trace-buffer-size 16384
```

After execution two files are produced next to the test binary
(inside `TEST_XDNA2/build_master/bin/`):

| File | Description |
|------|-------------|
| `trace.txt` | Raw hex trace words read back from the NPU |
| `trace.json` | Parsed trace in [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU) |

Open `trace.json` in [Perfetto UI](https://ui.perfetto.dev/) to visualise
core and memory trace timelines.

## How It Works

Enabling `--trace` triggers three additional MLIR code-transformation passes
during code generation:

1. **`MLIRCoreTracePass`** — Emits an `aie.trace` block on the compute tile
   configured for 8 core events (vector instructions, stalls, port activity)
   with packet-based routing.
2. **`MLIRMemTracePass`** — Emits a second `aie.trace` block for 8 memory/DMA
   events (S2MM/MM2S start, finish, starvation) with event-based
   start/stop synchronised to the core trace via broadcast signals.
3. **`MLIRTraceRuntimePass`** — Adds `trace_host_config` and
   `trace_start_config` calls to the runtime sequence to activate the
   configured traces at execution time.

On the host side, the XRT testbench (`main.cpp`) allocates a trace buffer
object, passes it as kernel argument 7, and writes the data back to `trace.txt` after execution.

The post-simulation callback in `deeployRunner_xdna2.py` then invokes the
mlir-aie trace parser (`aie.utils.trace.parse`) against the lowered MLIR
(`main_physical_with_elfs.mlir`) to produce the final `trace.json`.

## Traced Events

- `INSTR_EVENT_0`: Emitted by the `event0();` call, usually called at the beginning of the kernels (see `TargetLibraries/XDNA2/kernels/add.cc`).
- `INSTR_EVENT_1`: Emitted by the `event1();` call, usually called at the end of the kernels. 
- `INSTR_VECTOR`: Emitted every time the vector unit is used, can be useful to see how well the kernel is using the vector unit.
- `PORT_RUNNING_0`: Emitted when a DMA transfer is running on port 0.
- `PORT_RUNNING_1`: Emitted when a DMA transfer is running on port 1.
