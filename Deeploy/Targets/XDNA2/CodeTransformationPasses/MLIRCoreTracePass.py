# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass that emits a core trace configuration.

Emits an ``aie.trace`` block on the compute tile that captures core
instruction events, stall events, and port-monitoring events.  The
configuration name is appended to ``mlirBlock.traceConfigs`` so that the
runtime-sequence trace pass can activate it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from aie.dialects.aie import DMAChannelDir, TraceMode, TracePacketType, WireBundle, trace, trace_event, trace_mode, \
    trace_packet, trace_port, trace_start, trace_stop

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext

_DEFAULT_CORE_EVENTS = [
    "INSTR_EVENT_0",
    "INSTR_EVENT_1",
    "INSTR_VECTOR",
    "MEMORY_STALL",
    "STREAM_STALL",
    "LOCK_STALL",
    "PORT_RUNNING_0",
    "PORT_RUNNING_1",
]

_DEFAULT_CORE_PORTS = [
    (0, WireBundle.DMA, 0, DMAChannelDir.S2MM),
    (1, WireBundle.DMA, 0, DMAChannelDir.MM2S),
]


class MLIRCoreTracePass(MLIRCodeTransformationPass):
    """Emit a core trace configuration on the compute tile.

    Parameters
    ----------
    packetId : int
        Trace packet ID (default 1).
    events : list of str, optional
        Event names to capture (max 8).  Defaults to the reference set of
        instruction / stall / port-running events.
    ports : list of tuple, optional
        ``(slot, WireBundle, channel, DMAChannelDir)`` tuples for
        port-monitoring event slots.
    startBroadcast : int
        Broadcast channel that starts the trace (default 15).
    stopBroadcast : int
        Broadcast channel that stops the trace (default 14).
    """

    def __init__(
        self,
        packetId: int = 1,
        events: Optional[List[str]] = None,
        ports: Optional[List[tuple]] = None,
        startBroadcast: int = 15,
        stopBroadcast: int = 14,
    ) -> None:
        self.packetId = packetId
        self.events = events if events is not None else list(_DEFAULT_CORE_EVENTS)
        self.ports = ports if ports is not None else list(_DEFAULT_CORE_PORTS)
        self.startBroadcast = startBroadcast
        self.stopBroadcast = stopBroadcast

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        computeTile = mlirBlock.computeTile
        configName = f"core_trace_{name}"

        @trace(computeTile, configName)
        def _core_trace():
            trace_mode(TraceMode.EventTime)
            trace_packet(self.packetId, TracePacketType.Core)
            for event in self.events:
                trace_event(event)
            for slot, port, channel, direction in self.ports:
                trace_port(slot, port, channel, direction)
            trace_start(broadcast = self.startBroadcast)
            trace_stop(broadcast = self.stopBroadcast)

        mlirBlock.traceConfigs.append(configName)
        return ctxt, mlirBlock
