# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.Targets.Generic.TypeCheckers import ConvChecker
from Deeploy.Targets.Neureka.Templates.ConvTemplate import NeurekaDenseConv2D_Template, NeurekaDWConv2D_Template, \
    NeurekaPWConv2D_Template, NeurekaRqntDenseConv2D_Template, NeurekaRqntDWConv2D_Template, \
    NeurekaRqntPWConv2D_Template
from Deeploy.Targets.PULPOpen.Bindings import ClusterTransformer
from Deeploy.Targets.PULPOpen.TypeCheckers import PULPConvChecker

NeurekaRQSPWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaRqntPWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaPWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaPWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NeurekaRQSDWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaRqntDWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaDWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaDWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NeurekaRQSDenseConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaRqntDenseConv2D_Template,
        ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaDenseConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaDenseConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
