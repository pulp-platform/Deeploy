# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np
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

    @staticmethod
    def _isIntegerTensor(tensor: gs.Tensor) -> bool:
        dtype = getattr(tensor, "dtype", None)
        return dtype is not None and np.issubdtype(np.dtype(dtype), np.integer)

    def _hasSupportedTensorTypes(self, node: gs.Node) -> bool:
        tensors = list(node.inputs) + list(node.outputs)
        return all(self._isIntegerTensor(tensor) for tensor in tensors)

    def canExecute(self, node: gs.Node) -> bool:
        if not self._hasSupportedTensorTypes(node):
            return False
        return self.isPWConv(node) or self.isDWConv(node) or self.isDenseConv(node)
