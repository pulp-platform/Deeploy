# ----------------------------------------------------------------------
#
# File: Engine.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import DeploymentEngine, NodeMapper
from Deeploy.Targets.Generic.Layers import ConvLayer
from Deeploy.Targets.Neureka.Parsers import NeurekaDenseConv2DParser, NeurekaDWConv2DParser, NeurekaPWConv2DParser, \
    NeurekaRQSDenseConv2DParser, NeurekaRQSDWConv2DParser, NeurekaRQSPWConv2DParser
from Deeploy.Targets.Neureka.Tiler import NeurekaDenseConv2DTilingReadyBindings, NeurekaDWConv2DTilingReadyBindings, \
    NeurekaPWConv2DTilingReadyBindings, NeurekaRQSDenseConv2DTilingReadyBindings, \
    NeurekaRQSDWConv2DTilingReadyBindings, NeurekaRQSPWConv2DTilingReadyBindings, \
    NeurekaWmemDenseConv2DTilingReadyBindings, NeurekaWmemDWConv2DTilingReadyBindings, \
    NeurekaWmemPWConv2DTilingReadyBindings, NeurekaWmemRQSDenseConv2DTilingReadyBindings, \
    NeurekaWmemRQSDWConv2DTilingReadyBindings, NeurekaWmemRQSPWConv2DTilingReadyBindings
from Deeploy.Targets.PULPOpen.Layers import PULPRQSConvLayer

NeurekaRqntPWConv2DMapper = NodeMapper(
    NeurekaRQSPWConv2DParser(), NeurekaWmemRQSPWConv2DTilingReadyBindings + NeurekaRQSPWConv2DTilingReadyBindings)
NeurekaPWConv2DMapper = NodeMapper(NeurekaPWConv2DParser(),
                                   NeurekaWmemPWConv2DTilingReadyBindings + NeurekaPWConv2DTilingReadyBindings)

NeurekaRqntDWConv2DMapper = NodeMapper(
    NeurekaRQSDWConv2DParser(), NeurekaWmemRQSDWConv2DTilingReadyBindings + NeurekaRQSDWConv2DTilingReadyBindings)
NeurekaDWConv2DMapper = NodeMapper(NeurekaDWConv2DParser(),
                                   NeurekaWmemDWConv2DTilingReadyBindings + NeurekaDWConv2DTilingReadyBindings)

NeurekaRqntDenseConv2DMapper = NodeMapper(
    NeurekaRQSDenseConv2DParser(),
    NeurekaWmemRQSDenseConv2DTilingReadyBindings + NeurekaRQSDenseConv2DTilingReadyBindings)
NeurekaDenseConv2DMapper = NodeMapper(NeurekaDenseConv2DParser(),
                                      NeurekaWmemDenseConv2DTilingReadyBindings + NeurekaDenseConv2DTilingReadyBindings)

NeurekaMapping = {
    'RequantizedConv':
        PULPRQSConvLayer([NeurekaRqntPWConv2DMapper, NeurekaRqntDWConv2DMapper, NeurekaRqntDenseConv2DMapper]),
    'Conv':
        ConvLayer([NeurekaPWConv2DMapper, NeurekaDWConv2DMapper, NeurekaDenseConv2DMapper]),
}

_includeList = ["pulp_nnx_neureka.h", "pulp_nnx_util.h", "neureka_siracusa_bsp.h", "neureka.h", "neureka_task.h"]

_neurekaInitCode = r"""
neureka_siracusa_conf_t conf = {.max_stall = 8};
neureka_nnx_init(neureka_siracusa_get_dev(), &conf);
"""


class NeurekaEngine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = NeurekaMapping,
                 initCode: str = _neurekaInitCode,
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
