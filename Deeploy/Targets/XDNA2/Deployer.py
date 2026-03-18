# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 deployer — generates mlir-aie MLIR using ``aie.dialects``.

Unlike other Deeploy deployers that generate C code via Mako templates,
this deployer constructs an ``mlir.ir.Module`` with AIE dialect operations
and returns the verified MLIR text.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type

import onnx_graphsurgeon as gs

from aie.extras.context import mlir_mod_ctx
from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d
import aie.ir as ir

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import DeploymentPlatform, TopologyOptimizer
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MLIRDataTypes import MLIRNodeTemplate


class XDNA2Deployer(SignPropDeployer):
    """Deployer for the XDNA2 (AIE2p) platform.

    Generates an mlir-aie MLIR module by calling :meth:`emit` /
    :meth:`emitRuntimeSequence` on each bound :class:`MLIRNodeTemplate`.
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

        Iterates over bound layers, calls each template's ``emit()``
        to construct AIE operations, adds a ``runtime_sequence`` for
        host-side DMA, verifies the module, and returns the MLIR text.

        Returns
        -------
        str
            Verified MLIR module string.
        """
        assert self.prepared, "XDNA2Deployer.generateMLIR() called before prepare()"

        # Collect templates and their operator representations
        nodes = []
        for node_name, layer in self.layerBinding.items():
            mapper = layer.mapper
            template = mapper.binder.template
            op_repr = mapper.parser.operatorRepresentation

            if not isinstance(template, MLIRNodeTemplate):
                raise RuntimeError(
                    f"Node '{node_name}' has no MLIRNodeTemplate — "
                    f"only BF16 Add is supported in this release.")

            nodes.append((node_name, template, op_repr))

        if not nodes:
            raise RuntimeError("No bound layers found — cannot generate MLIR.")

        # Build the MLIR module
        with mlir_mod_ctx() as ctx:

            @aie_d.device(aie_d.AIEDevice.npu2)
            def _device():
                compute_tile = aie_d.tile(0, 2) # JUNGVI: This will have to change when we deploy on the whole array
                shim_tile = aie_d.tile(0, 0)

                # Emit each node's operations (ObjectFifos, core, kernel decls)
                for node_name, template, op_repr in nodes:
                    log.info(f"[XDNA2] Emitting MLIR for node '{node_name}'")
                    template.emit(op_repr,
                                  compute_tile=compute_tile,
                                  shim_tile=shim_tile) # JUNGVI: What should be the interface of the MLIR template emission exactly?

                # Runtime sequence: collect tensor types from all nodes' I/O
                # For now (single-node), derive from the first node.
                _, first_template, first_op_repr = nodes[0]
                params = first_template.getAIEParams(first_op_repr)
                num_elements = params['num_elements']
                tensor_ty = ir.MemRefType.get((num_elements,), ir.BF16Type.get())

                @aiex_d.runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
                def _seq(*args):
                    for _, template, op_repr in nodes:
                        template.emitRuntimeSequence(op_repr, list(args))

            module = ctx.module
            assert module.operation.verify(), \
                "[XDNA2] Generated MLIR module failed verification"

        mlir_str = str(module)
        log.info(f"[XDNA2] MLIR module generated ({len(mlir_str)} bytes)")
        return mlir_str
