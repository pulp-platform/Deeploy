# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Helper to enforce no-remainder (exact divisibility) tile constraints on XDNA2."""

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.TilingExtension.TilerModel import TilerModel


def addDivisibilityConstraints(tilerModel: TilerModel, bufferName: str, ctxt: NetworkContext) -> None:
    """Add ``fullDim == tileDimVar * quotientVar`` for every dimension of *bufferName*.

    This guarantees that tiles evenly divide each dimension — no remainder
    tiles — which is required by XDNA2 ObjectFifo-based DMA transfers.
    """
    shape = ctxt.lookup(bufferName).shape
    if isinstance(shape, int):
        shape = (shape,)

    for dimIdx, fullDim in enumerate(shape):
        tileDimVar = tilerModel.getTensorDimVar(tensorName = bufferName, dimIdx = dimIdx)
        quotientVar = tilerModel._addVariable(name = f"{bufferName}_divisibility_q_{dimIdx}",
                                              lowerBound = 1,
                                              upperBound = fullDim)
        tilerModel.addConstraint(fullDim == tileDimVar * quotientVar)
