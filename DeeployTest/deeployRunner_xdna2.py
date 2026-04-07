#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper that invokes the shared Deeploy test runner for the XDNA2 platform.

Usage (from DeeployTest/):
    python deeployRunner_xdna2.py -t Tests/Kernels/BF16/Add/Regular [--skipsim] [-v]
    python deeployRunner_xdna2.py -t Tests/Kernels/BF16/Add/Regular --trace [--trace-buffer-size 16384]
"""

import json
import os
import sys
from glob import glob

from testUtils.deeployRunner import main


def _add_xdna2_args(parser):
    """Register XDNA2-specific CLI arguments."""
    parser.add_argument('--trace', action = 'store_true', default = False,
                        help = 'Enable execution tracing in the generated MLIR')
    parser.add_argument('--trace-buffer-size', type = int, default = 8192,
                        help = 'Trace buffer size in bytes (default: 8192)')


def _add_xdna2_gen_args(args, gen_args_list):
    """Forward XDNA2-specific arguments to the generation script."""
    if getattr(args, 'trace', False):
        gen_args_list.append('--trace')
        trace_buffer_size = getattr(args, 'trace_buffer_size', 8192)
        if trace_buffer_size != 8192:
            gen_args_list.append(f'--trace-buffer-size={trace_buffer_size}')


def _xdna2_post_sim(config, result, args):
    """Parse trace.txt into a Perfetto-compatible trace.json after simulation."""
    if not getattr(args, 'trace', False):
        return

    build_dir = config.build_dir
    trace_txt = os.path.join(build_dir, "bin", "trace.txt")
    if not os.path.isfile(trace_txt):
        print(f"Warning: --trace enabled but {trace_txt} not found; skipping trace parsing.")
        return

    # Find the MLIR with lowered NpuWrite32 ops (trace event register config).
    # aiecc.py produces this when invoked with --dump-intermediates.
    prj_pattern = os.path.join(build_dir, "DeeployTest", "Platforms", "XDNA2",
                               "network.mlir.prj", "main_physical_with_elfs.mlir")
    candidates = glob(prj_pattern)
    if not candidates:
        print(f"Warning: lowered MLIR not found at {prj_pattern}; skipping trace parsing.")
        return
    lowered_mlir = candidates[0]

    trace_json = os.path.join(build_dir, "bin", "trace.json")

    try:
        from aie.utils.trace.parse import parse_mlir_trace_events, setup_trace_metadata, \
            convert_commands_to_json, check_for_valid_trace, trim_trace_pkts, \
            trace_pkts_de_interleave, convert_to_byte_stream, convert_to_commands, \
            align_column_start_index

        with open(trace_txt, "r") as f:
            trace_pkts = f.read().split("\n")

        with open(lowered_mlir, "r") as f:
            mlir_str = f.read()

        pid_events, events_module = parse_mlir_trace_events(mlir_str)

        if not check_for_valid_trace(trace_txt, trace_pkts):
            print(f"Warning: trace data in {trace_txt} appears invalid; skipping trace parsing.")
            return

        trimmed = trim_trace_pkts(trace_pkts)
        sorted_pkts = trace_pkts_de_interleave(trimmed)
        byte_streams = convert_to_byte_stream(sorted_pkts)
        commands = convert_to_commands(byte_streams, False)

        pid_events = align_column_start_index(pid_events, commands)

        trace_events = []
        setup_trace_metadata(trace_events, pid_events, events_module)
        convert_commands_to_json(trace_events, commands, pid_events, events_module)

        with open(trace_json, "w") as f:
            json.dump(trace_events, f)

        print(f"Trace parsed: {trace_json} ({len(trace_events)} events)")

    except SystemExit:
        print(f"Warning: trace parsing failed (mlir-aie parser error). "
              f"Ensure the build was done with --trace enabled.")
    except Exception as e:
        print(f"Warning: trace parsing failed: {e}")


if __name__ == '__main__':
    sys.exit(
        main(default_platform = "XDNA2",
             default_simulator = "host",
             tiling_enabled = True,
             parser_setup_callback = _add_xdna2_args,
             gen_args_callback = _add_xdna2_gen_args,
             post_sim_callback = _xdna2_post_sim))
