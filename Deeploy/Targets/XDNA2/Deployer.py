# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 deployer — generates mlir-aie MLIR using ``aie.dialects``.

Unlike other Deeploy deployers that generate C code via Mako templates,
this deployer constructs an ``mlir.ir.Module`` with AIE dialect operations
and returns the verified MLIR text.

MLIR generation is split into two phases orchestrated by
:class:`MLIRCodeTransformation`:

1. **Device phase** — inside ``@aie_d.device(npu2)``: for each operator,
   run ``devicePasses`` (ObjectFifo creation, external-kernel
   declaration) then call ``template.emit()`` (compute core only).
2. **Runtime-sequence phase** — inside ``@aiex_d.runtime_sequence``:
   for each operator, run ``runtimeSequencePasses`` (DMA configuration).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type

import aie.ir as ir
import onnx_graphsurgeon as gs
from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d
from aie.extras.context import mlir_mod_ctx

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import DeploymentPlatform, TopologyOptimizer
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MLIRDataTypes import MLIRCodeTransformation, MLIRExecutionBlock, MLIRNodeTemplate


class XDNA2Deployer(SignPropDeployer):
    """Deployer for the XDNA2 (AIE2p) platform.

    Generates an mlir-aie MLIR module via two-phase code transformation:

    * **Device phase**: ``MLIRObjectFifoPass`` creates ObjectFifos and
      declares external kernels; the bound ``MLIRNodeTemplate`` emits
      the compute core.
    * **Runtime-sequence phase**: ``MLIRRuntimeSequencePass`` configures
      shim DMA for L3 ↔ L1 transfers.

    The module is verified via MLIR's built-in verifier before being
    returned as a string.
    """

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = False,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets: Optional[Dict[str, int]] = None):
        super().__init__(
            graph,
            deploymentPlatform,
            inputTypes,
            loweringOptimizer,
            scheduler,
            name,
            default_channels_first = default_channels_first,
            deeployStateDir = deeployStateDir,
            inputOffsets = inputOffsets if inputOffsets is not None else {},
        )

    # ------------------------------------------------------------------
    # MLIR generation
    # ------------------------------------------------------------------

    def generateMLIR(self) -> str:
        """Generate an mlir-aie MLIR module for the prepared graph.

        Iterates over bound layers in two phases:

        1. **Device phase** — for each node, creates an
           :class:`MLIRExecutionBlock`, runs device-phase code-
           transformation passes (ObjectFifo creation, kernel
           declaration), then calls ``template.emit()`` (compute core).
        2. **Runtime-sequence phase** — opens an
           ``@aiex_d.runtime_sequence`` block, sets
           ``runtimeSequenceArgs`` on each block, then runs
           runtime-sequence passes (DMA configuration).

        Returns
        -------
        str
            Verified MLIR module string.
        """
        assert self.prepared, "XDNA2Deployer.generateMLIR() called before prepare()"

        # Collect per-node info from the bound layers
        nodes = []
        for nodeName, layer in self.layerBinding.items():
            mapper = layer.mapper
            binder = mapper.binder
            template = binder.template
            opRepr = mapper.parser.operatorRepresentation
            codeTransformer = binder.codeTransformer

            # Tiling constraint from the midend solver (may be None)
            executionBlock = binder.executionBlock
            tilingConstraint = getattr(executionBlock, 'patternMemoryConstraint', None)

            if not isinstance(template, MLIRNodeTemplate):
                raise RuntimeError(f"Node '{nodeName}' has no MLIRNodeTemplate — "
                                   f"only BF16 Add is supported in this release.")
            if not isinstance(codeTransformer, MLIRCodeTransformation):
                raise RuntimeError(f"Node '{nodeName}' uses a non-MLIR CodeTransformation — "
                                   f"expected MLIRCodeTransformation, got {type(codeTransformer).__name__}.")

            nodes.append({
                'nodeName': nodeName,
                'template': template,
                'opRepr': opRepr,
                'codeTransformer': codeTransformer,
                'tilingConstraint': tilingConstraint,
            })

        if not nodes:
            raise RuntimeError("No bound layers found — cannot generate MLIR.")

        # Build the MLIR module
        mlirBlocks = []

        with mlir_mod_ctx() as ctx:

            @aie_d.device(aie_d.AIEDevice.npu2)
            def _device():
                computeTile = aie_d.tile(0, 2)  # TODO: generalize to full array
                shimTile = aie_d.tile(0, 0)

                # === Device phase ===
                for node in nodes:
                    # Create MLIRExecutionBlock with deployer-level state
                    eb = MLIRExecutionBlock(computeTile = computeTile, shimTile = shimTile)
                    eb.operatorRepresentation = node['opRepr']
                    eb.patternMemoryConstraint = node['tilingConstraint']
                    eb.template = node['template']

                    log.info(f"[XDNA2] Device phase for '{node['nodeName']}'" +
                             (" (tiled)" if node['tilingConstraint'] else ""))

                    # Run device-phase passes:
                    #  1. MLIRObjectFifoPass — creates FIFOs, declares kernel
                    #  2. MLIRComputeCorePass — opens core + loops, calls
                    #     template.emit() with acquired FIFO elements in opRepr
                    self.ctxt, eb = node['codeTransformer'].applyDevicePasses(self.ctxt, eb, node['nodeName'])

                    mlirBlocks.append((node, eb))

                # === Runtime-sequence phase ===
                # Derive tensor type from the first node's numElements
                _, firstEb = mlirBlocks[0]
                numElements = firstEb.numElements
                tensorTy = ir.MemRefType.get((numElements,), ir.BF16Type.get())

                @aiex_d.runtime_sequence(tensorTy, tensorTy, tensorTy)
                def _seq(*args):
                    for node, eb in mlirBlocks:
                        eb.runtimeSequenceArgs = list(args)
                        log.info(f"[XDNA2] Runtime-sequence phase for '{node['nodeName']}'")
                        self.ctxt, eb = node['codeTransformer'].applyRuntimeSequencePasses(
                            self.ctxt, eb, node['nodeName'])

            module = ctx.module
            assert module.operation.verify(), \
                "[XDNA2] Generated MLIR module failed verification"

        mlirStr = str(module)
        log.info(f"[XDNA2] MLIR module generated ({len(mlirStr)} bytes)")
        return mlirStr
