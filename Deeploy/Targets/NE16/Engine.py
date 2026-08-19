# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import DeploymentEngine, NodeMapper
from Deeploy.Targets.Generic.Layers import ConvLayer
from Deeploy.Targets.NE16.Parsers import NE16DenseConv2DParser, NE16DWConv2DParser, NE16PWConv2DParser, \
    NE16RQSDenseConv2DParser, NE16RQSDWConv2DParser, NE16RQSPWConv2DParser
from Deeploy.Targets.NE16.Tiler import NE16DenseConv2DTilingReadyBindings, NE16DWConv2DTilingReadyBindings, \
    NE16PWConv2DTilingReadyBindings, NE16RQSDenseConv2DTilingReadyBindings, NE16RQSDWConv2DTilingReadyBindings, \
    NE16RQSPWConv2DTilingReadyBindings
from Deeploy.Targets.PULPOpen.Layers import PULPRQSConvLayer

NE16RqntPWConv2DMapper = NodeMapper(NE16RQSPWConv2DParser(), NE16RQSPWConv2DTilingReadyBindings)
NE16PWConv2DMapper = NodeMapper(NE16PWConv2DParser(), NE16PWConv2DTilingReadyBindings)

NE16RqntDWConv2DMapper = NodeMapper(NE16RQSDWConv2DParser(), NE16RQSDWConv2DTilingReadyBindings)
NE16DWConv2DMapper = NodeMapper(NE16DWConv2DParser(), NE16DWConv2DTilingReadyBindings)

NE16RqntDenseConv2DMapper = NodeMapper(NE16RQSDenseConv2DParser(), NE16RQSDenseConv2DTilingReadyBindings)
NE16DenseConv2DMapper = NodeMapper(NE16DenseConv2DParser(), NE16DenseConv2DTilingReadyBindings)

NE16Mapping = {
    'RequantizedConv': PULPRQSConvLayer([NE16RqntPWConv2DMapper, NE16RqntDWConv2DMapper, NE16RqntDenseConv2DMapper]),
    'Conv': ConvLayer([NE16PWConv2DMapper, NE16DWConv2DMapper, NE16DenseConv2DMapper]),
}

_includeList = ["pulp_nnx_ne16.h", "pulp_nnx_util.h", "ne16_pulp_bsp.h", "ne16.h", "ne16_task.h"]

_ne16InitCode = r"""
ne16_pulp_conf_t conf = {.max_stall = 8};
ne16_nnx_init(ne16_pulp_get_dev(), &conf);
"""


class NE16Engine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = NE16Mapping,
                 initCode: str = _ne16InitCode,
                 includeList: List[str] = _includeList,
                 enable3x3: bool = False,
                 enableStrides: bool = False) -> None:
        super().__init__(name, Mapping, initCode, includeList)

        self.enable3x3 = enable3x3
        self.enableStrides = enableStrides

    def isDenseConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            isinstance(node.inputs[1], gs.Constant) and \
            node.attrs['kernel_shape'] == [3, 3] and \
            node.attrs['dilations'] == [1, 1] and \
            node.attrs['group'] == 1 and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    def isPWConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            isinstance(node.inputs[1], gs.Constant) and \
            node.attrs['kernel_shape'] == [1, 1] and \
            node.attrs['dilations'] == [1, 1] and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    def isDWConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            isinstance(node.inputs[1], gs.Constant) and \
            node.attrs['kernel_shape'] == [3, 3] and \
            node.attrs['dilations'] == [1, 1] and \
            node.attrs['group'] != 1 and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    def canExecute(self, node: gs.Node) -> bool:
        if self.enable3x3:
            return self.isPWConv(node) or self.isDWConv(node) or self.isDenseConv(node)
        else:
            return self.isPWConv(node)
