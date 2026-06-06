# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import AddParser, DivParser, GEMMParser, MulParser, RQGEMMParser, \
    iHardswishParser, iRMSNormParser


class SnitchGEMMParser(GEMMParser):

    def parseNode(self, node: gs.Node) -> bool:
        ret = super().parseNode(node)

        if not ret:
            return False

        if not all([
                self.operatorRepresentation['transA'] == 0,
        ]):
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        if not all([
                self.operatorRepresentation['batch'] == 1,
        ]):
            return ctxt, False

        return newCtxt, True


class SnitchRQGEMMParser(RQGEMMParser):

    def parseNode(self, node: gs.Node) -> bool:
        ret = super().parseNode(node)

        if not ret:
            return False

        if not all([
                self.operatorRepresentation['transA'] == 0,
        ]):
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        if not all([
                self.operatorRepresentation['batch'] == 1,
        ]):
            return ctxt, False

        return newCtxt, True


class SnitchRMSNormParser(iRMSNormParser):
    """FP32 RMSNorm parser. Inherits parseNodeCtxt from iRMSNormParser."""

    def parseNode(self, node: gs.Node) -> bool:
        if node.op != 'RMSNorm':
            return False
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            return False

        eps = node.attrs.get('eps', node.attrs.get('epsilon', 1e-6))
        self.operatorRepresentation['eps'] = f"{float(eps):.10e}f"

        stash_type = node.attrs.get('stash_type', 1)
        if stash_type != 1:
            raise ValueError(f"RMSNorm: only stash_type=1 (FP32) is supported, got {stash_type}")

        return True


class SnitchHardSwishParser(iHardswishParser):
    """FP32 HardSwish parser. Inherits parseNodeCtxt from iHardswishParser."""

    def parseNode(self, node: gs.Node) -> bool:
        if node.op != 'HardSwish':
            return False
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            return False
        return True


class SnitchAddParser(AddParser):
    """Inherits from Generic AddParser which already handles broadcasting."""

    pass


class _ScalarElementwiseMixin:
    """Shared parsing for FP32 Div/Mul on Snitch.

    The kernels (Div_fp32/Mul_fp32) only support equal-shape element-wise
    operation or a scalar second operand (they read input2[0]); there is no
    broadcasting kernel. Reject any genuine broadcast so unsupported shapes
    fail to bind instead of generating out-of-bounds reads.
    """

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if not ret:
            return ctxt, False

        shape1 = list(ctxt.lookup(node.inputs[0].name).shape)
        shape2 = list(ctxt.lookup(node.inputs[1].name).shape)

        second_is_scalar = (np.prod(shape2) == 1)
        if shape1 != shape2 and not second_is_scalar:
            return ctxt, False

        self.operatorRepresentation['size'] = int(np.prod(shape1))
        self.operatorRepresentation['is_scalar'] = second_is_scalar

        return ctxt, True


class SnitchDivParser(_ScalarElementwiseMixin, DivParser):
    pass


class SnitchMulParser(_ScalarElementwiseMixin, MulParser):
    pass
