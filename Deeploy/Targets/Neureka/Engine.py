# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import DeploymentEngine, NodeMapper
from Deeploy.Targets.Generic.Layers import ConvLayer
from Deeploy.Targets.Neureka.Parsers import NeurekaDenseConv2DParser, NeurekaDWConv2DParser, NeurekaPWConv2DParser, \
    NeurekaRQSDenseConv2DParser, NeurekaRQSDWConv2DParser, NeurekaRQSPWConv2DParser
from Deeploy.Targets.Neureka.Tiler import NeurekaDenseConv2DTilingReadyBindings, NeurekaDWConv2DTilingReadyBindings, \
    NeurekaPWConv2DTilingReadyBindings, NeurekaRQSDenseConv2DTilingReadyBindings, \
    NeurekaRQSDWConv2DTilingReadyBindings, NeurekaRQSPWConv2DTilingReadyBindings
from Deeploy.Targets.PULPOpen.Layers import PULPRQSConvLayer

NeurekaRqntPWConv2DMapper = NodeMapper(NeurekaRQSPWConv2DParser(), NeurekaRQSPWConv2DTilingReadyBindings)
NeurekaPWConv2DMapper = NodeMapper(NeurekaPWConv2DParser(), NeurekaPWConv2DTilingReadyBindings)

NeurekaRqntDWConv2DMapper = NodeMapper(NeurekaRQSDWConv2DParser(), NeurekaRQSDWConv2DTilingReadyBindings)
NeurekaDWConv2DMapper = NodeMapper(NeurekaDWConv2DParser(), NeurekaDWConv2DTilingReadyBindings)

NeurekaRqntDenseConv2DMapper = NodeMapper(NeurekaRQSDenseConv2DParser(), NeurekaRQSDenseConv2DTilingReadyBindings)
NeurekaDenseConv2DMapper = NodeMapper(NeurekaDenseConv2DParser(), NeurekaDenseConv2DTilingReadyBindings)

NeurekaMapping = {
    'RequantizedConv':
        PULPRQSConvLayer([NeurekaRqntPWConv2DMapper, NeurekaRqntDWConv2DMapper, NeurekaRqntDenseConv2DMapper]),
    'Conv':
        ConvLayer([NeurekaPWConv2DMapper, NeurekaDWConv2DMapper, NeurekaDenseConv2DMapper]),
}

_includeList = [
    "pulp_nnx_neureka.h", "pulp_nnx_util.h", "neureka_siracusa_bsp.h", "neureka.h", "neureka_task.h", "neureka_gvsoc.h"
]

_neurekaInitCode = r"""
neureka_siracusa_conf_t conf = {.max_stall = 8};
neureka_nnx_init(neureka_siracusa_get_dev(), &conf);
// neureka_gvsoc_log_activate(neureka_siracusa_get_dev(), NEUREKA_GVSOC_LOG_LEVEL_ALL, NEUREKA_GVSOC_LOG_FORMAT_HEXADECIMAL);
"""


class NeurekaEngine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = NeurekaMapping,
                 initCode: str = _neurekaInitCode,
                 includeList: List[str] = _includeList,
                 enableStrides: bool = False) -> None:
        super().__init__(name, Mapping, initCode, includeList)

        self.enableStrides = enableStrides

    @staticmethod
    def _isSupportedConvNode(node: gs.Node) -> bool:
        # Common N-EUREKA preconditions for all convolution flavors. Keep this
        # structural: engine coloring runs before Deeploy has reliable type info.
        return node.op in ["Conv", "RequantizedConv"] and \
            len(node.inputs) > 1 and \
            isinstance(node.inputs[1], gs.Constant) and \
            node.attrs.get('dilations') == [1, 1]

    def _hasSupportedStrides(self, node: gs.Node) -> bool:
        # Strided convolutions are opt-in because not every N-EUREKA setup enables
        # them, while unit strides are always supported.
        return node.attrs.get('strides') == [1, 1] or self.enableStrides

    def isDenseConv(self, node) -> bool:
        return self._isSupportedConvNode(node) and \
            node.attrs.get('kernel_shape') == [3, 3] and \
            node.attrs.get('group', 1) == 1 and \
            self._hasSupportedStrides(node)

    def isPWConv(self, node) -> bool:
        return self._isSupportedConvNode(node) and \
            node.attrs.get('kernel_shape') == [1, 1] and \
            self._hasSupportedStrides(node)

    def isDWConv(self, node) -> bool:
        return self._isSupportedConvNode(node) and \
            node.attrs.get('kernel_shape') == [3, 3] and \
            node.attrs.get('group', 1) != 1 and \
            self._hasSupportedStrides(node)

    def canExecute(self, node: gs.Node) -> bool:
        return self.isPWConv(node) or self.isDWConv(node) or self.isDenseConv(node)
