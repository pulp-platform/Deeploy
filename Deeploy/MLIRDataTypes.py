# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Base class for MLIR-emitting node templates.

This module provides :class:`MLIRNodeTemplate`, a :class:`NodeTemplate`
subclass whose ``generate()`` method produces an MLIR string instead of C
code.  Concrete subclasses override :meth:`emit` to populate an
``mlir.ir.Module`` using dialect-specific Python bindings (e.g.
``aie.dialects`` for the XDNA2 backend).

The class is intentionally dialect-agnostic so that future MLIR-based
backends (NVGPU, Linalg, …) can reuse the same base.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from Deeploy.DeeployTypes import NodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation


class MLIRNodeTemplate(NodeTemplate):
    """NodeTemplate subclass that emits MLIR instead of C code.

    Subclasses must override :meth:`emit` to add dialect operations to an
    ``mlir.ir.Module`` (or region / insertion point provided via *kwargs*).

    ``generate()`` is overridden as a convenience that constructs a
    standalone module, calls :meth:`emit`, and returns the MLIR text.
    The base-class ``alignToContext`` / ``hoistTransientBuffers`` hooks are
    retained and work unchanged.
    """

    def __init__(self):
        # Empty Mako template — no C code is generated.
        super().__init__("")

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Populate an MLIR module with the operations for this node.

        The caller (typically the deployer) sets up an ``mlir.ir.Module``
        with the appropriate device wrapper and passes dialect-specific
        context through *kwargs* (e.g. insertion point, tile references,
        ObjectFifo handles).

        Parameters
        ----------
        operatorRepresentation : OperatorRepresentation
            The parser's node representation (buffer names, sizes, types …).
        **kwargs
            Dialect-specific context provided by the deployer.
        """
        ...

    # ------------------------------------------------------------------
    # NodeTemplate overrides
    # ------------------------------------------------------------------

    def generate(self, operatorRepresentation={}, **kwargs) -> str:
        """Generate an MLIR string for this node.

        This default implementation is a thin wrapper: it delegates to
        :meth:`emit`.  Deployers that need to build a single module from
        multiple nodes should call :meth:`emit` directly with the shared
        module context and then stringify the complete module themselves.

        Returns
        -------
        str
            MLIR text (printable module or fragment).
        """
        self.emit(operatorRepresentation, **kwargs)
        return ""
