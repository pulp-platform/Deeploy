# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime-sequence pass that activates trace configurations.

Emits ``aie.trace.host_config`` to set up the host-side trace buffer and
``aie.trace.start_config`` for each trace configuration registered by
device-phase passes on the :class:`MLIRExecutionBlock`.

This pass must run **before** the DMA configuration pass
(:class:`MLIRRuntimeSequencePass`) inside the ``runtime_sequence`` block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from aie.dialects.aie import trace_host_config, trace_start_config

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


class MLIRTraceRuntimePass(MLIRCodeTransformationPass):
    """Emit trace host configuration and activate trace configs.

    Reads ``mlirBlock.traceConfigs`` (populated by device-phase trace
    passes such as :class:`MLIRCoreTracePass` / :class:`MLIRMemTracePass`)
    and ``mlirBlock.traceBufferSize`` (set by the deployer).  If there are
    no trace configs, this pass is a no-op.
    """

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:

        if not mlirBlock.traceConfigs:
            return ctxt, mlirBlock

        trace_host_config(buffer_size = mlirBlock.traceBufferSize)

        for configName in mlirBlock.traceConfigs:
            trace_start_config(configName)

        return ctxt, mlirBlock
