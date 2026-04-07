# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass that emits a memory trace configuration.

Emits an ``aie.trace`` block on the compute tile that captures DMA
start/finish/starvation events from the memory module trace unit.  The
configuration name is appended to ``mlirBlock.traceConfigs`` so that the
runtime-sequence trace pass can activate it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from aie.dialects.aie import (TracePacketType, trace, trace_event, trace_packet, trace_start, trace_stop)

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext

_DEFAULT_MEM_EVENTS = [
    "DMA_S2MM_0_START_TASK",
    "DMA_S2MM_1_START_TASK",
    "DMA_MM2S_0_START_TASK",
    "DMA_S2MM_0_FINISHED_TASK",
    "DMA_S2MM_1_FINISHED_TASK",
    "DMA_MM2S_0_FINISHED_TASK",
    "DMA_S2MM_0_STREAM_STARVATION",
    "DMA_S2MM_1_STREAM_STARVATION",
]


class MLIRMemTracePass(MLIRCodeTransformationPass):
    """Emit a memory trace configuration on the compute tile.

    Parameters
    ----------
    packetId : int
        Trace packet ID (default 3).
    events : list of str, optional
        Event names to capture (max 8).  Defaults to DMA start / finish /
        starvation events matching the reference example.
    startEvent : str
        Event that starts the trace (default ``"BROADCAST_15"``).
    stopEvent : str
        Event that stops the trace (default ``"BROADCAST_14"``).
    """

    def __init__(
        self,
        packetId: int = 3,
        events: Optional[List[str]] = None,
        startEvent: str = "BROADCAST_15",
        stopEvent: str = "BROADCAST_14",
    ) -> None:
        self.packetId = packetId
        self.events = events if events is not None else list(_DEFAULT_MEM_EVENTS)
        self.startEvent = startEvent
        self.stopEvent = stopEvent

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        computeTile = mlirBlock.computeTile
        configName = f"mem_trace_{name}"

        @trace(computeTile, configName)
        def _mem_trace():
            trace_packet(self.packetId, TracePacketType.Mem)
            for event in self.events:
                trace_event(event)
            trace_start(event = self.startEvent)
            trace_stop(event = self.stopEvent)

        mlirBlock.traceConfigs.append(configName)
        return ctxt, mlirBlock
