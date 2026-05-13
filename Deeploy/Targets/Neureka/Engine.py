# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List

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

    @staticmethod
    def _isIntegerDtype(dtype: Any) -> bool:
        # ONNX-GraphSurgeon may expose either numpy dtypes/classes or plain ONNX
        # enum integers depending on the graph loading path. Treat unknown or
        # missing dtypes as inconclusive instead of rejecting the node outright.
        if dtype is None:
            return False

        try:
            return np.issubdtype(dtype, np.integer)
        except TypeError:
            return dtype in {
                2,  # UINT8
                3,  # INT8
                4,  # UINT16
                5,  # INT16
                6,  # INT32
                7,  # INT64
                12,  # UINT32
                13,  # UINT64
            }

    @classmethod
    def _hasIntegerTensorType(cls, node: gs.Node) -> bool:
        # Prefer real tensor metadata when it is available. This catches already
        # integer-typed ONNX graphs without depending on exporter-specific names.
        return any(cls._isIntegerDtype(getattr(tensor, "dtype", None)) for tensor in [*node.inputs, *node.outputs])

    @staticmethod
    def _hasQuantizedProvenance(node: gs.Node) -> bool:
        # Some quantized Deeploy/PACT graphs still carry FLOAT ONNX annotations
        # before type inference. In that case the stable signal is the provenance
        # left by quantization/integerization passes on tensors or attributes.
        quantizedMarkers = ("INTEGERIZE", "QUANT", "REQUANT", "PACT")
        names = [getattr(tensor, "name", "") for tensor in [*node.inputs, *node.outputs]]
        attrNames = [str(name) for name in node.attrs.keys()]
        return any(marker in name.upper() for name in [*names, *attrNames] for marker in quantizedMarkers)

    @classmethod
    def _hasNeurekaCompatibleSemantics(cls, node: gs.Node) -> bool:
        # RequantizedConv is produced only after a quantized convolution/requant
        # pattern was merged, so its op already carries the integer semantics that
        # N-EUREKA expects. Plain Conv needs an additional signal; otherwise FP32
        # convolutions from mixed models would be colored for N-EUREKA and fail
        # later in parsing/binding.
        if node.op == "RequantizedConv":
            return True

        return cls._hasIntegerTensorType(node) or cls._hasQuantizedProvenance(node)

    def isDenseConv(self, node) -> bool:
        return self._isSupportedConvNode(node) and \
            self._hasNeurekaCompatibleSemantics(node) and \
            node.attrs.get('kernel_shape') == [3, 3] and \
            node.attrs.get('group', 1) == 1 and \
            self._hasSupportedStrides(node)

    def isPWConv(self, node) -> bool:
        return self._isSupportedConvNode(node) and \
            self._hasNeurekaCompatibleSemantics(node) and \
            node.attrs.get('kernel_shape') == [1, 1] and \
            self._hasSupportedStrides(node)

    def isDWConv(self, node) -> bool:
        return self._isSupportedConvNode(node) and \
            self._hasNeurekaCompatibleSemantics(node) and \
            node.attrs.get('kernel_shape') == [3, 3] and \
            node.attrs.get('group', 1) != 1 and \
            self._hasSupportedStrides(node)

    def canExecute(self, node: gs.Node) -> bool:
        # Engine coloring runs before Deeploy type inference, and ONNX dtype
        # annotations are not reliable for every quantized graph. Still, N-EUREKA
        # is an integer accelerator, so the coloring must avoid pure FP Conv
        # nodes and let the fallback engine handle them.
        return self.isPWConv(node) or self.isDWConv(node) or self.isDenseConv(node)
